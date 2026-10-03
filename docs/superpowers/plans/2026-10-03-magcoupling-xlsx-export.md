# Magcoupling GUI: spreadsheet download of the current layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A download button that saves the magnetic coupling calculator's current design as an .xlsx workbook laid out like the page (a Summary, the Inputs in the order shown, the Results by physics chain, the Assumptions), values only.

**Architecture:** A pure layout module (`gui::spreadsheet`) turns the design shown (a `Snapshot`: the inputs shown, their results, the sizing state, the input order, the share link, the export time) into sheets of rows of cells with a style tag, reusing the page's own code rather than copying its rules: the inputs side's groups, section order and headings (`InputCatalogue::groups_in`, `InputGroup::plain_sections` and `advanced_sections`, `InputSection::heading_in`, which Task 2 adds and the panel's inputs side then calls too), the results table's grouped lines and badges (`table_lines`, `entry_level`, `worst_level`), `dashboard_lines`, the equation registry's `render::plain`, `assumptions::states`; a thin writer (`gui::xlsx`) renders those sheets with rust_xlsxwriter. The panel's two buttons (the header's "Export spreadsheet", the results table's "Export XLSX") call one method that queues `PanelRequest::SaveFile`, whose `contents` becomes `Vec<u8>` so both hosts (the standalone app's `app::files`, the linkage app's calculator window) save binary files.

**Tech Stack:** Rust 2024, egui 0.32 (headless tests), rust_xlsxwriter 0.99.1 (new, pinned; its `wasm` feature on wasm32 only), calamine 0.36 and zip 8 (dev-dependencies, to read the file back in tests), wasm-bindgen 0.2.114 (unchanged), Playwright for the browser check.

**Spec:** the user's request of 2026-10-03, "a download button for the spreadsheet equivalent of the current layout of the website", clarified the same day: "No need for live formulas from excel back to site" (a values snapshot of the current design laid out like the site; nothing reads a spreadsheet back). The layout it mirrors is the M4 layout of `docs/superpowers/specs/2026-09-28-magcoupling-calculator-design.md` as the ordering plan left it (`docs/superpowers/plans/2026-10-02-magcoupling-gui-ordering.md`, decisions O-1 to O-8). This plan's Decisions to confirm (below) settle what the request leaves open.

**Execution worktree:** `C:/Users/Cole/source/repos/lsim-mag-xlsx`, branch `magcoupling/xlsx`, created by Task 0 from `main` at `49fbd3f` with LF line endings and a worktree-scoped `core.autocrlf=false`. Every command uses absolute paths into it. Nothing is pushed.

**Process (the user's lean process):** five tasks (0 to 4). Tasks 1 to 4 give the exact code and text, so the implementers transcribe; every implementation and every per-task review runs on `sonnet`; there is no pre-flight scan, so every block below quotes `main` at `49fbd3f` (or the file as the earlier tasks leave it) exactly, and was parsed back out of this file and replayed literally (Verification record); the whole-branch review after Task 4 runs on the session model.

## Decisions to confirm

**Confirmed (user, 2026-10-03):** X-1 (values only: no formulas, no round trip from Excel back to the site).

**Not yet confirmed.** These choices arose while turning the request into code. The plan implements the recommended option of each (the code and docs cite the ids). Task 0 Step 7 asks the user and records the answers; if the user picks another option, the task that implements it stops and escalates instead of improvising.

| Id | Question | Recommended (implemented here) | Alternatives | Tasks |
|---|---|---|---|---|
| X-1 | What the file holds | **Values only** (confirmed): every cell a number or a text, no formula; nothing reads a spreadsheet back into the site | (b) live formulas built from the equation records (declined by the user) | 2, 3 |
| X-2 | Which sheets, in which order | **Summary, Inputs, Results, Assumptions** | (b) no Assumptions sheet (the Inputs sheet flags the 15 assumption inputs); (c) one sheet per physics chain; (d) the Results before the Inputs | 2 |
| X-3 | The order of the Inputs sheet | **The order the inputs side shows when the button is pressed:** the workflow groups (the default) or, after the toggle, the workbook groups; each group's heading, its section headings, and in the workflow order its Advanced heading before its advanced sections, laid out by the page's own `InputGroup::plain_sections`, `advanced_sections` and `InputSection::heading_in` (the inputs side calls them too); the inputs filter is ignored (every input is exported) | (b) always the workflow order; (c) always the workbook order (the original workbook's sheets); (d) both orders, two sheets | 2, 4 |
| X-4 | The Key design group | **Not repeated:** a "Key design" flag column marks its ten inputs, so every input is on exactly one row | (b) a Key design block on top, as on the page (ten inputs listed twice) | 2 |
| X-5 | The order of the Results sheet | **Always by physics chain, as the results table groups them by default** (the table's own `table_lines`, every row, every group open: the headline, the eight chains, "Other results" by package), each heading with the worst level of its checks; the CSV and JSON exports keep the engine's order | (b) follow the table's order toggle; (c) the engine's order | 2 |
| X-6 | The share link | **Text, labelled "Share link (paste into a browser)":** the link names every input and is 2,510 characters at the default design, over Excel's 2,080-character hyperlink limit (rust_xlsxwriter refuses a longer link) | (b) a hyperlink when the link fits, text otherwise (in practice always text); (c) leave it out | 2, 3 |
| X-7 | The file name | **`magcoupling-results.xlsx`**: the stem of the CSV and JSON exports beside its button | (b) `magcoupling-design.xlsx`; (c) a dated name, `magcoupling-2026-10-03-1432.xlsx` (unique per export) | 3, 4 |
| X-8 | Where the button goes | **Both:** "Export spreadsheet" in the header after "Copy share link" (always in view) and "Export XLSX" beside "Export CSV" and "Export JSON" in the results table; one panel method behind both | (b) the results table only; (c) the header only | 4 |
| X-9 | The xlsx library | **rust_xlsxwriter `=0.99.1`** (pure Rust, Excel-valid files, builds for wasm32), its `wasm` feature on wasm32 only (without it the workbook's creation time calls `SystemTime::now()`, which panics in the browser); gate 11 checks both lock files; its zip 8 brings zopfli and switches flate2 to its zlib-rs backend in every build that carries the panel (share links, the calculator's and the linkage app's, change their bytes; a link written before still loads, pinned by a test in each crate); measured web bundle growth: magcoupling-web +1.05 MiB (5,118,840 to 6,219,376 bytes, +21.5 %; gzip -9 +381,658 bytes), linkage-web +1.02 MiB (12,115,195 to 13,189,224 bytes, +8.9 %; gzip -9 +383,970 bytes) (Verification record) | (b) a hand-written writer: a zip of the OOXML parts with flate2 (already in both bundles, and its `Crc`): no new dependency and almost none of that growth, but about 300 lines of format code ours to keep Excel-valid (Task 3 re-derived); (c) umya-spreadsheet (reads and writes; heavier) | 3 |
| X-10 | How binary content reaches the hosts | **`PanelRequest::SaveFile.contents` becomes `Vec<u8>`** (one variant; the text exports pass `.into_bytes()`); the standalone app writes the bytes (`app::files::write_file`), the linkage window downloads them (`export::download::download_bytes`) | (b) a second variant `SaveBytes` beside `SaveFile` (two arms in each host); (c) an enum `FileContents { Text, Bytes }` | 1 |
| X-11 | The check colours and the headings | **The page's badge colours as fills**, taken from `Level::color(&egui::Visuals::dark())` (green 5AC878, amber FF8F00, red FF0000; the standalone page forces egui's dark visuals, so this is a fixed palette: the linkage app's calculator window may draw in light visuals, whose green and amber badges are darker, and the fills do not follow them), bold black text, the badge's word ("OK", "Check", "Fails") in the Check column; a check greyed by the end effect has none. A group heading (an input group, a result chain, "Other results", a Summary block) bold 12 pt on a light grey (E7E6E6), so it stands apart from the bold 11 pt section and Advanced headings inside it. The Inputs, Results and Assumptions sheets freeze their header row and carry Excel's filter buttons on it; the Summary does neither (it is read top to bottom, and its dashboard header sits below a number of rows that depends on the sizing mode and the banners) | (b) Excel's built-in Good, Neutral and Bad styles (pale fills); (c) coloured text, no fill; (d) the first draft: no grey, no filters, every sheet's first row frozen (the Summary's title alone) | 2, 3 |
| X-12 | The Summary's header rows | **The panel's heading as the title, the export time in UTC (an Excel date), "magcoupling-rs 0.1.0"**; no design name: the app has none, and adding one would change the design file format | (b) a design-name input saved in design files and share links (a `DESIGN_VERSION` question); (c) the browser's local time | 2, 3 |

## Global Constraints

- Base: `main` at `49fbd3f` ("docs(ai): next items, the physics walkthrough and the coupler landing page (user requests)"). Worktree `C:/Users/Cole/source/repos/lsim-mag-xlsx`, branch `magcoupling/xlsx`. Never modify `C:/Users/Cole/source/repos/linkage_simulation` itself beyond the `worktree add` of Task 0 Step 1.
- The engine and its data are untouched: no file under `magcoupling-rs/src/engine/`, `magcoupling-rs/tests/` or `reference/` is edited; every engine and integration test keeps Task 0's count.
- Design files and the CSV and JSON exports keep their bytes: the text exports only gain `.into_bytes()` on the way to the host (Task 1). Share links are compressed by flate2's zlib-rs backend once Task 3 adds rust_xlsxwriter (X-9) and still decode: at the default design the calculator's payload differs from the one `49fbd3f` writes (both 2,462 characters), and so does a linkage link; Task 3 pins a link each app wrote at `49fbd3f` and checks that it still loads.
- Dependencies: rust_xlsxwriter exactly `=0.99.1` (feature `gui`; its `wasm` feature in the wasm32 target table only), js-sys added to feature `gui` (wasm32 only; the existing optional dependency), calamine `0.36` and zip `8` (`default-features = false`, `features = ["deflate"]`) as dev-dependencies (cargo resolves calamine, zip and the crates they bring to the newest compatible versions when Task 3 runs; the versions quoted there are for information: an exact pin of the dev-dependencies would not reach `linkage-sim-rs/Cargo.lock`, which does not resolve another crate's dev-dependencies). After Task 3 both `Cargo.lock` files hold `rust_xlsxwriter` `0.99.1` and still `wasm-bindgen` `0.2.114` (gate 11, which Task 3 extends to `rust_xlsxwriter`). Never run a broad `cargo update`.
- `workbook-parity` never reaches a shipped build (gate 10 is unchanged and must stay green).
- Formatting: `cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-xlsx/magcoupling-rs/Cargo.toml --check` prints nothing after every task (every block is already in rustfmt's layout). Never run `cargo fmt` in `linkage-sim-rs` (not rustfmt-clean at `49fbd3f`); the linkage blocks are in rustfmt's layout where they add code.
- Lints: magcoupling-rs clippy with warnings as errors stays clean (gates 5, 8 and 9: `--all-targets`, `--all-targets --features app`, wasm32 `--features gui --lib` and the web binary).
- UI text is ASCII (`->`, never an arrow glyph): `every_text_the_panel_shows_has_glyphs_in_the_default_fonts` checks the new button texts.
- Blocks: they quote the files with LF line endings, as git stores them (this machine's system gitconfig sets `core.autocrlf=true`; Task 0 makes the worktree LF). "Create `path`:" writes a new file with the block's text and a final newline. "In `path`, replace: ... with: ..." is one exact replacement: the old block occurs exactly once in the file, as whole lines (a block may start with an empty line). "In `path`, replace (part of one line): ... with: ..." replaces text inside one long line (the README's table rows, the one-line entries of `docs/ai/*.yaml`), also occurring exactly once; its new text may end that line and add lines after it. Apply a step's blocks in the order given. If an old block is not found, stop and escalate; never improvise a match.
- No test in this plan reads a source file; a test that did would normalize CRLF (`.replace("\r\n", "\n")`).
- Gate runs (Task 0 and every task before its commit): `MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-xlsx/linkage-sim-rs/scripts/gate.sh` must end `GATE PASS` with no `SKIP gate` line; then `git -C C:/Users/Cole/source/repos/lsim-mag-xlsx checkout -- docs/chebyshev_lambda` (the linkage tests rewrite those PNGs; never commit them).
- Windows paths: the worktree path is short on purpose (a build under a deep directory fails with `LNK1104`, MAX_PATH).
- Commits: one per task, subjects `feat(magcoupling-rs): ...` (Task 4's carries the docs and this plan). Every commit message ends with exactly two lines: a `Co-Authored-By:` line naming the model that writes the commit, then `Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9`. The blocks write `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` (the plan's writer); a `sonnet` implementer writes its own model's name there. Nothing is pushed.
- Docs change with code (repo rule, `docs/ai/01-meta.yaml`; the user's rule): Task 3 updates the README's feature table with the dependency it adds; Task 4 updates `magcoupling-rs/README.md`, `docs/ai/02-system.yaml`, `03-structure.yaml`, `04-memory.yaml` (it resolves the open item "NEXT (user request 2026-10-03): a download button for the spreadsheet ...") and `05-update-tracker.md`, and commits this plan as `docs/superpowers/plans/2026-10-03-magcoupling-xlsx-export.md`. Read `docs/ai/*.yaml` first (Task 0).
- Model tiers (CLAUDE.md section 5): every task gives the exact code, so the implementations and the per-task reviews run on `sonnet` (`model: 'sonnet'`); Task 0 is commands and reporting (`sonnet`). A `sonnet` attempt that ends `blocked`, leaves a check red or is rejected in review escalates every retry to the session model. The whole-branch review after Task 4 runs on the session model (omit `model`, comment `// session model: final whole-branch review`). No task is physics: the engine is untouched and the spreadsheet only lays out values the page already shows.

## Review Focus

The conditions the request implies but does not name that are most likely to bite a user, each pinned by a test in the task that owns the code:

1. **A result that is not a finite number** (`+inf` with no hot limit, `NaN` with no torque at it: E20's positive beta without a grade). Expected: the cell holds the text the CSV writes (`+inf`, `-inf`, `NaN`); no number cell holds a non-finite value and the export never fails. Tests: `a_number_that_is_not_finite_is_the_csv_s_text` (Task 2), `the_workbook_reads_back_cell_for_cell` with that design (Task 3).
2. **The share link is longer than Excel takes as a hyperlink** (2,510 characters at the default design; Excel's limit is 2,080). Expected: the link is written as text the user pastes into a browser, and the export succeeds. Test: `a_text_longer_than_a_cell_holds_is_cut_to_excel_s_limit_saying_how_long_it_was` writes a 2,536-character link and reads it back whole (Task 3).
3. **A text input longer than a cell holds** (design files of up to 1 MiB load; Excel's cell limit is 32,767 characters as Excel counts them, in UTF-16 code units, so an emoji counts two; rust_xlsxwriter refuses a string over 32,767 `char`s and so lets a text of emoji through). Expected: the text is cut after the last whole character that fits, ending with how long it was, and the export succeeds. Test: the same Task 3 test, with a 40,000-character part name and 20,000 emoji (40,000 UTF-16 units), and a text at the limit and one unit over.
4. **A design outside the end-effect model's range** (f_end <= 0: the page greys the pull-out and the numbers computed from it). Expected: a check greyed on the page has no level in the Results sheet (the hot minimum is not red), the Summary carries the end-effect banner in red and notes the greyed dashboard rows. Test: `a_check_carries_its_level_and_a_greyed_check_none` (Task 2).
5. **The browser.** Expected: the button works on the web, where `Workbook::new()` would call `SystemTime::now()` and panic without rust_xlsxwriter's `wasm` feature, and the export time comes from `js_sys::Date`. No headless test can run wasm32 code: Task 3 Step 5 checks with `cargo tree` that the feature is on for wasm32 and off natively, and Task 4 Step 7 downloads the file from `/magcoupling/` in a browser and opens it.
6. **A share link written before this branch** (rust_xlsxwriter's zip switches flate2 to its zlib-rs backend, so the bytes of a new link differ). Expected: a link written at `49fbd3f` still opens its design, in both apps. Tests: `a_share_link_written_before_the_zlib_rs_backend_still_loads` (magcoupling) and `share_url_written_before_the_zlib_rs_backend_still_decodes` (linkage), Task 3.

Also covered: Torque -> Magnets (the Inputs sheet holds the solved value, noted, and the share link reopens the sizing mode: `torque_to_magnets_shows_the_sizing_and_notes_the_free_variable_s_row`, `the_spreadsheet_follows_the_input_order_shown_and_torque_to_magnets`), the workbook order toggle (the same tests and `the_inputs_follow_the_order_shown_under_their_group_section_and_advanced_headings`, which checks the inputs in exactly the order the page draws them and a heading for each section but the group's own), the page's own order and badges shared rather than copied (`a_group_lists_its_sections_and_their_headings_as_the_page_draws_them`, `a_row_carries_its_check_s_level_and_a_heading_the_worst_of_its_rows`, and the Results sheet checked against `table_lines`' headings), an empty text input such as a part name left out for manual magnets (noted blank: `an_input_row_flags_changes_assumptions_key_design_and_notes_choices_and_blanks`), a write error (reported in the header instead of a request: `export_spreadsheet`'s `Err` arm, reached only by a failure of rust_xlsxwriter itself, which the cut texts above rule out for the known limits).

## File Structure

Created (under `C:/Users/Cole/source/repos/lsim-mag-xlsx/`):

| File | Responsibility | Task |
|---|---|---|
| `magcoupling-rs/src/gui/spreadsheet.rs` | the spreadsheet's layout: `Snapshot`, `sheets(&Snapshot) -> Vec<Sheet>`, `Sheet` (with its frozen rows and filter flag), `Cell`, `CellValue`, `CellStyle`, `value_cell`, `level_cell`, the sheet names and column headers, the Summary's labels; its tests | 2 |
| `magcoupling-rs/src/gui/xlsx.rs` | the writer: `spreadsheet_bytes`, `xlsx_bytes` (rust_xlsxwriter), `cell_text`, `fill`, `GROUP_FILL`, `now_unix_s`, `XLSX_FILE_NAME`, `XLSX_MIME`, `MAX_CELL_CHARS`, `TIME_FORMAT`; its tests (read back with calamine and zip) | 3 |

Modified:

| File | Change | Task |
|---|---|---|
| `magcoupling-rs/src/gui/panel.rs` | `SaveFile.contents: Vec<u8>` and `.into_bytes()` at its three producers (1); `assumptions_banner` shared with the Summary, and the inputs side drawing its sections with the inputs module's helpers (2); `EXPORT_SPREADSHEET` in the header, `export_spreadsheet`, the results table's `ExportXlsx` arm (4); tests | 1, 2, 4 |
| `magcoupling-rs/src/app/files.rs` | `save` takes bytes; `write_file` (native); a test | 1 |
| `linkage-sim-rs/src/gui/export/download.rs` | `write_file` after the native dialog; a test | 1 |
| `linkage-sim-rs/src/gui/calculator_window.rs` | `download_bytes` (1); the Excel workbook save filter and its test row (4) | 1, 4 |
| `magcoupling-rs/src/gui/corrections.rs` | `marks_values` made `pub(crate)` | 2 |
| `magcoupling-rs/src/gui/inputs.rs` | `InputGroup::plain_sections`, `advanced_sections`, `has_advanced` and `InputSection::heading_in` (the page's section order and headings, shared by the inputs side, the filter's runs and the spreadsheet); a test | 2 |
| `magcoupling-rs/src/gui/results_table.rs` | `RUST_ONLY`, `entry_level` (a row's badge, which the table's rows now call), `worst_level` through it; a test (2); `EXPORT_XLSX`, `TableAction::ExportXlsx` and its button (4) | 2, 4 |
| `magcoupling-rs/src/gui/mod.rs` | `pub mod spreadsheet;` (2), `pub mod xlsx;` (3) | 2, 3 |
| `magcoupling-rs/src/gui/test_support.rs` | `read_xlsx` (calamine), `xlsx_part` (zip) | 3 |
| `magcoupling-rs/src/gui/session.rs`, `linkage-sim-rs/src/gui/state/file_io.rs` | a test each: a share link `49fbd3f` wrote still decodes after the zlib-rs backend | 3 |
| `magcoupling-rs/Cargo.toml`, `magcoupling-rs/Cargo.lock`, `linkage-sim-rs/Cargo.lock` | rust_xlsxwriter (and js-sys) in feature `gui`, calamine and zip dev-dependencies; both lock files updated by cargo | 3 |
| `linkage-sim-rs/scripts/gate.sh` | gate 11 checks `rust_xlsxwriter` too | 3 |
| `magcoupling-rs/README.md` | the feature table (3); the export, the module list, the test table, the linkage window (4) | 3, 4 |
| `docs/ai/02-system.yaml`, `03-structure.yaml`, `04-memory.yaml`, `05-update-tracker.md` | invariants, lessons, structure, the open item resolved, the tracker entry | 4 |
| `docs/superpowers/plans/2026-10-03-magcoupling-xlsx-export.md` | this plan, committed | 4 |

## Verification record

This plan was replayed before it was handed over, twice: once for the first draft and again, from scratch, after the critic's findings were fixed (the Self-review record lists the fixes). This record is the second replay's. The code was developed in a scratch git repository holding an LF export of `49fbd3f` (`git -c core.autocrlf=false archive 49fbd3f`), one commit per task; this file's blocks were generated from that history (each hunk's old text widened until it occurs exactly once, as whole lines; a hunk that reaches into a file's test module belongs to the task's Step 1, the rest to its Step 3), and the docs blocks are the exact replacements applied there. The 77 blocks were then **parsed back out of this file and applied literally, task by task and step by step, to a fresh LF export of `49fbd3f`** (a scratch git repository at `X:/verify`, reset to its base commit with its build caches kept, one commit per task; `X:` is a `subst` drive for `C:/Users/Cole/AppData/Local/Temp/claude/C--Users-Cole-source-repos-linkage-simulation/3165685f-bf49-42bb-a036-671c0e065429/scratchpad/xlsx-plan`, for MAX_PATH), each step's commands run as written with the worktree path replaced by the scratch path. Every old block occurred exactly once. After every task the replayed files equal the development tree's file for file, both `Cargo.lock` files included (regenerated by cargo, not by a block). After the replay only prose changed in this file (the expected outputs and these records); the blocks are identical, checked by parsing both versions.

- **Task 0:** magcoupling with `app`: library `497 passed`, every integration binary as listed in Step 5; linkage library `955 passed`, every binary as listed; the gate passed (`GATE PASS`, no `SKIP gate`); `wasm-bindgen 0.2.114`, `rustc 1.89.0`.
- **Red steps failed as stated** (the exact lines are in each Step 2) and **green steps passed** with the counts given. After every task `cargo fmt --check` printed nothing, magcoupling clippy with warnings as errors ended `Finished`, and the gate passed (`GATE PASS`, no `SKIP gate`), so magcoupling clippy on every target (native, with `app`, wasm32 `gui` and the web binary), both wasm32 checks, the workbook-parity guard with its four negative controls (the four shipped builds: magcoupling-rs features `app, default, gui` for the calculator's two and `default, gui` for the linkage app's two), the lock parity (after Task 3 `rust_xlsxwriter 0.99.1 in both lock files` and still `wasm-bindgen 0.2.114`), the linkage tests (the calculator window's included) and the oracle stayed green throughout.

| After task | magcoupling `--features app --lib` | linkage `--lib` | Red step (Step 2) | Green step (Step 4) |
|---|---|---|---|---|
| 0 (base) | 497 | 955 | — | — |
| 1 | 498 | 956 | 6 errors (magcoupling), 2 (linkage) | 4 and 26 passed |
| 2 | 512 | 956 | 290 errors | 14 passed |
| 3 | 518 | 957 | 37 errors | 5 and 1 passed (the linkage link test: 1 passed, Step 5) |
| 4 | 520 | 957 | 4 errors; the filter test failed | 14 and 1 passed |

- **Share links across the backend switch:** the default design's payload written at `49fbd3f` (miniz_oxide) and after Task 3 (zlib-rs) are both 2,462 characters and differ from the fourth character on; a linkage link of the same mechanism JSON differs too (from the 27th). Both pinned tests decode the `49fbd3f` payloads after Task 3, and gate 1 and gate 7 run them.
- **Task 4 Step 7, on the replayed tree:** `build_web.sh` through both guards (`linkage-web builds without workbook-parity`, `magcoupling-web builds without workbook-parity`, `Build complete!` and `Magcoupling build complete!`). Neither bundle holds the text `SystemTime::now() is before Unix epoch`; both hold rust_xlsxwriter's. Sizes against the same scripts on a fresh export of `49fbd3f`: `linkage-web_bg.wasm` 12,115,195 to 13,189,224 bytes (+1,074,029, +8.9 %; GNU gzip -9 4,326,801 to 4,710,771, +383,970), `magcoupling-web_bg.wasm` 5,118,840 to 6,219,376 bytes (+1,100,536, +21.5 %; gzip -9 2,040,984 to 2,422,642, +381,658); the first replay's gzip figures came from another compressor, so only these pairs compare. Served with `python -m http.server 8093` and driven with Python Playwright (the installed Chrome) at 1400 x 900: `/magcoupling/` logged `magcoupling explorer: 392 equations`; the header showed "Export spreadsheet" after "Copy share link"; a click on it downloaded `magcoupling-results.xlsx` and the status line said so. openpyxl 3.1.2 read the four sheets; the title; the export time as an Excel date (`yyyy-mm-dd hh:mm:ss`, 2026-10-03 21:17:21 UTC); the share link as plain text, no hyperlink (2,499 characters, the page's own address); 172 inputs and 1,086 results, each once (the Inputs sheet 215 rows: the header, 8 group headings, 34 section and Advanced headings and the 172 inputs; the Results 1,107: the header, 20 headings and the 1,086 results); no formula, no text that reads as a number, no number that is not finite; numbers as numbers (the face gap 1.4 flagged Key design, the pull-out 2.6884762950539796 with `E3 E7 E8`); the clearance screening's "Fails" in bold on FFFF0000 and the end-effect check's "OK" on FF5AC878; f_end's equation `f_{end} = 1 − c_{end} · (τ_p/L)`; every group heading (the 8 input groups, the 10 result headings from "Headline" to "Other results", the Summary's two blocks) bold 12 pt on FFE7E6E6 and every section, Advanced and package heading bold 11 pt with no fill; the Inputs, Results and Assumptions sheets frozen at A2 with `<autoFilter ref="A1:J215"/>`, `A1:H1107` and `A1:H16` (and their hidden `_xlnm._FilterDatabase` names), the Summary neither; the document title "Magnetic coupling calculator". The link from the file, opened in the browser, reopened the design (`magcoupling: loaded the design from the share link (sizing: Magnets -> Torque)`). In the linkage app (`/?tool=magcoupling`) the calculator window's header button and its results table's "Export XLSX" (wrapped onto the table's second toolbar row, beside the rows' badges) each downloaded a workbook equal to the standalone one, sheet for sheet and cell for cell but the export time and the link's address, through the linkage download helper. Every page logged 0 errors and 0 warnings. Both servers' Python processes were stopped by their PIDs and the ports checked free.
- **The window that moved (first replay: "not reproduced"), now pinned down:** a scripted instant click (Playwright's `mouse.click`: move, press and release at once) on the calculator window's "Export spreadsheet" moved the window about 80 points left and 52 down, every time; a click that hovers first and holds the press 100 ms, as a person clicks, left it in place, and "Save design" (whose frame is far shorter than the one that writes the workbook) left it in place under the same instant click, in this build and in the base build. Not investigated further (no code change here touches the window); the user's hand check before merge covers it.
- **Mutation checks** (the replayed tree, each restored after; each failed the named test): a section's inputs in reverse, the critic's mutation (`the_inputs_follow_the_order_shown_under_their_group_section_and_advanced_headings`); the group's own section given a heading (the same test, and `a_group_lists_its_sections_and_their_headings_as_the_page_draws_them`); the Advanced flag always blank (`the_inputs_follow_...`); an infinite number written as a number (`a_number_that_is_not_finite_is_the_csv_s_text`); every row's level red, greyed checks included (`a_check_carries_its_level_and_a_greyed_check_none`); every row's level dropped (`a_row_carries_its_check_s_level_and_a_heading_the_worst_of_its_rows`); an empty text input not noted (`an_input_row_flags_changes_assumptions_key_design_and_notes_choices_and_blanks`); the Summary frozen (`the_workbook_has_the_summary_inputs_results_and_assumptions_sheets`); no cut of overlong text, and a cut that counts characters instead of UTF-16 units (`a_text_longer_than_a_cell_holds_is_cut_to_excel_s_limit_saying_how_long_it_was`); no filters, and no grey on the group headings (`the_file_has_frozen_headers_filters_widths_bold_headings_fills_and_no_formula`); the panel exporting its inputs instead of the design shown (`the_spreadsheet_follows_the_input_order_shown_and_torque_to_magnets`); the Key design rows repeated above the groups (`every_input_is_on_one_row_with_its_value_unit_and_cell_in_either_order`); the fills from egui's light visuals (`the_level_fills_are_the_page_s_badge_colours`).

Not exercised by the replay: the native apps by hand (their save goes through rfd's dialog; `write_file`, which writes the bytes after it, is tested in both hosts), opening the file in Excel or LibreOffice (openpyxl and calamine read it; the user's check before merge), and the user's answers to the Decisions to confirm.

---

## Tasks

### Task 0: Worktree, baseline and context

**Model:** `sonnet` (creating the worktree, running commands and reporting output; CLAUDE.md section 5). Step 7 is the controller's.

Verification only; nothing to write.

**Files:**
- none

**Interfaces:**
- Consumes: `main` at `49fbd3f`, or a later commit that leaves the files this plan's blocks touch unchanged (Step 2).
- Produces: the worktree `C:/Users/Cole/source/repos/lsim-mag-xlsx` on the new branch `magcoupling/xlsx` with LF line endings, a green baseline with its test counts, and the controller's record of the user's answers to the Decisions to confirm.

- [ ] **Step 1: Create the worktree**

Run:

```bash
git -C C:/Users/Cole/source/repos/linkage_simulation log --oneline -1 main
git -C C:/Users/Cole/source/repos/linkage_simulation config extensions.worktreeConfig true
git -C C:/Users/Cole/source/repos/linkage_simulation -c core.autocrlf=false worktree add -b magcoupling/xlsx C:/Users/Cole/source/repos/lsim-mag-xlsx main
git -C C:/Users/Cole/source/repos/lsim-mag-xlsx config --worktree core.autocrlf false
git -C C:/Users/Cole/source/repos/lsim-mag-xlsx rev-parse --abbrev-ref HEAD
git -C C:/Users/Cole/source/repos/lsim-mag-xlsx status --short
```

Expected: the `log` line is `49fbd3f docs(ai): next items, the physics walkthrough and the coupler landing page (user requests)` or a later commit; `worktree add` prints `Preparing worktree (new branch 'magcoupling/xlsx')`; then `magcoupling/xlsx`; no status lines (the main checkout's untracked `docs/analyses/2026-05-28-press-4bar-analysis.md` is not in the worktree).

- [ ] **Step 2: Check that the blocks' files are still those of `49fbd3f`**

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-xlsx diff --stat 49fbd3f HEAD -- magcoupling-rs linkage-sim-rs/src/gui/calculator_window.rs linkage-sim-rs/src/gui/export/download.rs linkage-sim-rs/src/gui/state/file_io.rs linkage-sim-rs/scripts/gate.sh linkage-sim-rs/Cargo.toml linkage-sim-rs/Cargo.lock docs/ai
```

Expected: no output. If any file is listed, stop and escalate: the blocks that touch it must be re-derived first.

- [ ] **Step 3: Check the line endings**

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-xlsx config --get core.autocrlf; head -c 3000 C:/Users/Cole/source/repos/lsim-mag-xlsx/magcoupling-rs/src/gui/panel.rs | tr -cd '\r' | wc -c
```

Expected: `false` and `0` carriage returns. If the count is not 0: run `git -C C:/Users/Cole/source/repos/lsim-mag-xlsx rm -rq --cached . && git -C C:/Users/Cole/source/repos/lsim-mag-xlsx reset -q --hard` and check again.

- [ ] **Step 4: Read the project context**

Read `C:/Users/Cole/source/repos/lsim-mag-xlsx/docs/ai/01-meta.yaml`, `02-system.yaml`, `03-structure.yaml`, `04-memory.yaml` (the item this plan resolves: "NEXT (user request 2026-10-03): a download button for the spreadsheet ..."); `magcoupling-rs/README.md` ("Features and binaries", "The panel", the test table, "In the linkage app (M5)"); and the code this plan changes or reads: `magcoupling-rs/src/gui/panel.rs` (`PanelRequest`, `view_ui`, `header_ui`, the test `Harness`), `results_table.rs` (`TableAction`, `ResultsTable::ui`, `row_ui`, `table_entries`, `exact_number`, `table_lines`, `Line`, `ResultOrder`), `result_groups.rs` (`result_groups`), `inputs.rs` (`InputCatalogue::groups_in`, `InputGroup`, `InputSection`, `SectionMatches::heading`, `KEY_DESIGN`, `ADVANCED_HEADING`), `session.rs` (`encode_share_payload`, `decode_share_payload`), linkage `gui/state/file_io.rs` (`encode_mechanism_for_url`, `decode_mechanism_from_url`), `dashboard.rs` (`Level`, `dashboard_lines`, `check_level`, `warning_lines`, `end_effect_banner`), `corrections.rs`, `readouts.rs` (`registry`), `engine/explain/render.rs` (`plain`), `engine/assumptions.rs` (`states`), `app/files.rs`, `app.rs` (`MagcouplingApp::ui`), and `linkage-sim-rs/src/gui/calculator_window.rs` (`CalculatorWindow::show`, `save_filter`) and `export/download.rs`; and this plan's Decisions to confirm.

- [ ] **Step 5: Run both crates' tests**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-xlsx/magcoupling-rs/Cargo.toml --features app 2>&1 | grep -E "Running|test result"
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-xlsx/linkage-sim-rs/Cargo.toml --all 2>&1 | grep -E "Running|test result"
git -C C:/Users/Cole/source/repos/lsim-mag-xlsx checkout -- docs/chebyshev_lambda
```

Expected: magcoupling with `app`: the library `test result: ok. 497 passed`; the binaries `magcoupling_app` and `magcoupling_web` 0; `tests\assumptions.rs` 8; `deviations` 54; `differential` 19; `explain` 12 (2 ignored); `grades` 11; `material_library` 4; `material_links` 11; `parity` 4; `python_schema` 7; `robustness` 12; `schema` 7; `sizing` 31; `static_data` 8; doc-tests 1 (1 ignored). Then the linkage library `test result: ok. 955 passed`; the binaries `linkage_gui` and `linkage_web` 0; `tests\actuator_force_label.rs` 4; `braindump_repro` 1; `compound_actuator_rebuild` 5; `compound_force_integration` 9; `dxf_import_test` 1; `force_zone_tests` 8; `geometry_tests` 18; `golden_fixtures` 11; `gravity_breakdown_reference` 2; `mount_point_integration` 1; `parallelogram_actuator_sample` 4; `property_tests` 8; `singular_behavior` 18; doc-tests 0. If a count differs but every binary is `ok`, record the actual counts and read every later count as an offset from them.

- [ ] **Step 6: Check the oracle Python, the toolchain and the gate**

Run:

```bash
ls C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe
wasm-bindgen --version
rustc --version
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-xlsx/linkage-sim-rs/scripts/gate.sh 2>&1 | tail -n 3
git -C C:/Users/Cole/source/repos/lsim-mag-xlsx checkout -- docs/chebyshev_lambda
```

Expected: the Python path is listed; `wasm-bindgen 0.2.114`; `rustc 1.89.0` or later (rust_xlsxwriter 0.99.1 and calamine 0.36 need 1.88); the gate's last line `GATE PASS`, and no `SKIP gate` line in its output.

- [ ] **Step 7: Confirm the decisions (controller)**

Show the user the Decisions to confirm table (X-1 is confirmed; ask about X-2 to X-12) and record the answers in the controller's notes. If the user picks another option for any id, stop before the task that implements it (the table's last column) and escalate: its blocks must be re-derived.

### Task 1: Binary files for the hosts (`SaveFile` carries bytes)

**Model:** `sonnet` (exact code; CLAUDE.md section 5). Escalate a retry to the session model.

**Files:**
- Modify: `magcoupling-rs/src/gui/panel.rs` (`PanelRequest::SaveFile.contents: Vec<u8>`; `.into_bytes()` in `view_ui`'s CSV and JSON arms and `header_ui`'s Save design; three test expectations)
- Modify: `magcoupling-rs/src/app/files.rs` (`save` takes `&[u8]`; `write_file`; a test)
- Modify: `linkage-sim-rs/src/gui/export/download.rs` (`write_file`; a test)
- Modify: `linkage-sim-rs/src/gui/calculator_window.rs` (`download_bytes`)

**Interfaces:**
- Consumes: `linkage-sim-rs` `export::download::download_bytes(default_filename: &str, mime_type: &str, contents: &[u8], filter: FileFilter) -> DownloadOutcome` (exists).
- Produces: `PanelRequest::SaveFile { file_name: String, mime: &'static str, contents: Vec<u8> }`; `magcoupling::app::files::save(file_name: &str, mime: &str, contents: &[u8]) -> Option<Result<String, String>>`; private `write_file(path: &std::path::Path, contents: &[u8])` in both hosts (`-> Result<String, String>` in magcoupling-rs, `-> DownloadOutcome` in linkage-sim-rs). `MagcouplingApp::ui` is unchanged (`&contents` coerces to `&[u8]`).

- [ ] **Step 1: Write the failing tests**

In `magcoupling-rs/src/gui/panel.rs`, replace:

````rust
                PanelRequest::SaveFile {
                    file_name: "magcoupling-results.csv".to_owned(),
                    mime: "text/csv",
                    contents: results_csv(&results),
                },
                PanelRequest::SaveFile {
                    file_name: "magcoupling-results.json".to_owned(),
                    mime: "application/json",
                    contents: results_json(&harness.panel.design(), &results),
                },
            ]
        );
````

with:

````rust
                PanelRequest::SaveFile {
                    file_name: "magcoupling-results.csv".to_owned(),
                    mime: "text/csv",
                    contents: results_csv(&results).into_bytes(),
                },
                PanelRequest::SaveFile {
                    file_name: "magcoupling-results.json".to_owned(),
                    mime: "application/json",
                    contents: results_json(&harness.panel.design(), &results).into_bytes(),
                },
            ]
        );
````

In `magcoupling-rs/src/gui/panel.rs`, replace:

````rust
                PanelRequest::SaveFile {
                    file_name: DESIGN_FILE_NAME.to_owned(),
                    mime: "application/json",
                    contents: design_to_json(&gap_design(1.41)),
                },
                PanelRequest::OpenDesign,
            ]
````

with:

````rust
                PanelRequest::SaveFile {
                    file_name: DESIGN_FILE_NAME.to_owned(),
                    mime: "application/json",
                    contents: design_to_json(&gap_design(1.41)).into_bytes(),
                },
                PanelRequest::OpenDesign,
            ]
````

In `magcoupling-rs/src/gui/panel.rs`, replace:

````rust
        let [PanelRequest::SaveFile { contents, .. }] = &requests[..] else {
            panic!("one export: {requests:?}")
        };
        let json: serde_json::Value = serde_json::from_str(contents).unwrap();
        let design = design_from_json(&json["design"].to_string()).unwrap();
        assert_eq!(design.inputs, harness.panel.shown_inputs());
        assert_eq!(&compute_all(&design.inputs), harness.panel.results());
````

with:

````rust
        let [PanelRequest::SaveFile { contents, .. }] = &requests[..] else {
            panic!("one export: {requests:?}")
        };
        let text = std::str::from_utf8(contents).expect("the JSON export is UTF-8");
        let json: serde_json::Value = serde_json::from_str(text).unwrap();
        let design = design_from_json(&json["design"].to_string()).unwrap();
        assert_eq!(design.inputs, harness.panel.shown_inputs());
        assert_eq!(&compute_all(&design.inputs), harness.panel.results());
````

In `magcoupling-rs/src/app/files.rs`, replace:

````rust
        }
    }
}
````

with:

````rust
        }
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn a_saved_file_holds_the_bytes_as_they_are() {
        // A zip archive's first bytes, then bytes that are no UTF-8 text and line breaks: the
        // file holds them unchanged.
        let contents = [0x50, 0x4B, 0x03, 0x04, 0xFF, 0x00, 0xFE, b'\n', b'\r'];
        let path =
            std::env::temp_dir().join(format!("magcoupling-files-test-{}.bin", std::process::id()));
        let message = write_file(&path, &contents).expect("written");
        assert_eq!(message, format!("Saved {}", path.display()));
        assert_eq!(std::fs::read(&path).unwrap(), contents);
        std::fs::remove_file(&path).unwrap();
        // A folder that does not exist: the reason, not a panic.
        let missing = std::env::temp_dir()
            .join("magcoupling-no-such-folder")
            .join("a.xlsx");
        let error = write_file(&missing, &contents).unwrap_err();
        assert!(
            error.starts_with(&format!("Could not save {}: ", missing.display())),
            "{error}"
        );
    }
}
````

In `linkage-sim-rs/src/gui/export/download.rs`, replace:

````rust
    DownloadOutcome::Saved(format!("Downloaded: {}", default_filename))
}
````

with:

````rust
    DownloadOutcome::Saved(format!("Downloaded: {}", default_filename))
}

#[cfg(all(test, feature = "native"))]
mod tests {
    use super::*;

    #[test]
    fn a_saved_file_holds_the_bytes_as_they_are() {
        // A zip archive's first bytes (the calculator's spreadsheet export), then bytes that are
        // no UTF-8 text and line breaks: the file holds them unchanged.
        let contents = [0x50, 0x4B, 0x03, 0x04, 0xFF, 0x00, 0xFE, b'\n', b'\r'];
        let path =
            std::env::temp_dir().join(format!("linkage-download-test-{}.bin", std::process::id()));
        let DownloadOutcome::Saved(message) = write_file(&path, &contents) else {
            panic!("not saved");
        };
        assert_eq!(message, format!("Saved: {}", path.display()));
        assert_eq!(std::fs::read(&path).unwrap(), contents);
        std::fs::remove_file(&path).unwrap();
        // A folder that does not exist: the reason, not a panic.
        let missing = std::env::temp_dir()
            .join("linkage-no-such-folder")
            .join("a.xlsx");
        let DownloadOutcome::Failed(error) = write_file(&missing, &contents) else {
            panic!("a missing folder cannot be written");
        };
        assert!(error.starts_with("Write failed: "), "{error}");
    }
}
````


- [ ] **Step 2: Run the tests to see them fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-xlsx/magcoupling-rs/Cargo.toml --features app --lib -- files:: the_export_buttons save_design_queues json_export_in_torque 2>&1 | grep -E "^error" | sort | uniq -c
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-xlsx/linkage-sim-rs/Cargo.toml --lib download:: 2>&1 | grep -E "^error" | sort | uniq -c
```

Expected: both fail to compile their tests, magcoupling-rs with these lines (the four `mismatched types` are the expectations' `.into_bytes()` against `contents: String`):

```
      1 error: could not compile `magcoupling-rs` (lib test) due to 6 previous errors; 1 warning emitted
      4 error[E0308]: mismatched types
      2 error[E0425]: cannot find function `write_file` in this scope
```

and linkage-sim-rs with these (its 22 warnings are the linkage tests' own, at `49fbd3f` too):

```
      1 error: could not compile `linkage-sim-rs` (lib test) due to 2 previous errors; 22 warnings emitted
      2 error[E0425]: cannot find function `write_file` in this scope
```


- [ ] **Step 3: Make the hosts take bytes**

In `magcoupling-rs/src/gui/panel.rs`, replace:

````rust
        file_name: String,
        /// The media type (the web download's Blob type).
        mime: &'static str,
        contents: String,
    },
    /// Let the user pick a design file, then hand its text to
    /// [`MagcouplingPanel::load_design_file`].
````

with:

````rust
        file_name: String,
        /// The media type (the web download's Blob type).
        mime: &'static str,
        /// The file's bytes: UTF-8 text for a design file and the CSV and JSON exports, as
        /// they are for a binary format.
        contents: Vec<u8>,
    },
    /// Let the user pick a design file, then hand its text to
    /// [`MagcouplingPanel::load_design_file`].
````

In `magcoupling-rs/src/gui/panel.rs`, replace:

````rust
                    Some(TableAction::ExportCsv) => self.requests.push(PanelRequest::SaveFile {
                        file_name: CSV_FILE_NAME.to_owned(),
                        mime: "text/csv",
                        contents: results_csv(&self.results),
                    }),
                    Some(TableAction::ExportJson) => self.requests.push(PanelRequest::SaveFile {
                        file_name: JSON_FILE_NAME.to_owned(),
````

with:

````rust
                    Some(TableAction::ExportCsv) => self.requests.push(PanelRequest::SaveFile {
                        file_name: CSV_FILE_NAME.to_owned(),
                        mime: "text/csv",
                        contents: results_csv(&self.results).into_bytes(),
                    }),
                    Some(TableAction::ExportJson) => self.requests.push(PanelRequest::SaveFile {
                        file_name: JSON_FILE_NAME.to_owned(),
````

In `magcoupling-rs/src/gui/panel.rs`, replace:

````rust
                                sizing: self.sizing,
                            },
                            &self.results,
                        ),
                    }),
                    None => {}
                }
````

with:

````rust
                                sizing: self.sizing,
                            },
                            &self.results,
                        )
                        .into_bytes(),
                    }),
                    None => {}
                }
````

In `magcoupling-rs/src/gui/panel.rs`, replace:

````rust
                self.requests.push(PanelRequest::SaveFile {
                    file_name: DESIGN_FILE_NAME.to_owned(),
                    mime: "application/json",
                    contents: design_to_json(&design),
                });
            }
            if ui.button(LOAD_DESIGN).clicked() {
````

with:

````rust
                self.requests.push(PanelRequest::SaveFile {
                    file_name: DESIGN_FILE_NAME.to_owned(),
                    mime: "application/json",
                    contents: design_to_json(&design).into_bytes(),
                });
            }
            if ui.button(LOAD_DESIGN).clicked() {
````

In `magcoupling-rs/src/app/files.rs`, replace:

````rust
//! (rfd); on the web a browser download, and rfd's file picker. The same pattern as
//! `linkage-sim-rs/src/gui/export/download.rs`.

/// Saves `contents`: natively to the file the user picks, on the web as a download. Returns
/// the message to show, or `None` when the user cancelled.
pub fn save(file_name: &str, mime: &str, contents: &str) -> Option<Result<String, String>> {
    save_impl(file_name, mime, contents)
}

#[cfg(not(target_arch = "wasm32"))]
fn save_impl(file_name: &str, _mime: &str, contents: &str) -> Option<Result<String, String>> {
    let extension = file_name.rsplit_once('.').map_or("", |(_, ext)| ext);
    let path = rfd::FileDialog::new()
        .set_file_name(file_name)
        .add_filter(extension, &[extension])
        .save_file()?;
    Some(
        std::fs::write(&path, contents)
            .map(|()| format!("Saved {}", path.display()))
            .map_err(|error| format!("Could not save {}: {error}", path.display())),
    )
}

#[cfg(target_arch = "wasm32")]
fn save_impl(file_name: &str, mime: &str, contents: &str) -> Option<Result<String, String>> {
    Some(download(file_name, mime, contents.as_bytes()).map(|()| format!("Downloaded {file_name}")))
}

/// A browser download: a Blob behind a transient `<a download>` that is clicked.
````

with:

````rust
//! (rfd); on the web a browser download, and rfd's file picker. The same pattern as
//! `linkage-sim-rs/src/gui/export/download.rs`.

/// Saves `contents` (text or binary, as the bytes are): natively to the file the user picks, on
/// the web as a download. Returns the message to show, or `None` when the user cancelled.
pub fn save(file_name: &str, mime: &str, contents: &[u8]) -> Option<Result<String, String>> {
    save_impl(file_name, mime, contents)
}

#[cfg(not(target_arch = "wasm32"))]
fn save_impl(file_name: &str, _mime: &str, contents: &[u8]) -> Option<Result<String, String>> {
    let extension = file_name.rsplit_once('.').map_or("", |(_, ext)| ext);
    let path = rfd::FileDialog::new()
        .set_file_name(file_name)
        .add_filter(extension, &[extension])
        .save_file()?;
    Some(write_file(&path, contents))
}

/// Writes `contents` to `path` byte for byte; the message to show, or why it failed.
#[cfg(not(target_arch = "wasm32"))]
fn write_file(path: &std::path::Path, contents: &[u8]) -> Result<String, String> {
    std::fs::write(path, contents)
        .map(|()| format!("Saved {}", path.display()))
        .map_err(|error| format!("Could not save {}: {error}", path.display()))
}

#[cfg(target_arch = "wasm32")]
fn save_impl(file_name: &str, mime: &str, contents: &[u8]) -> Option<Result<String, String>> {
    Some(download(file_name, mime, contents).map(|()| format!("Downloaded {file_name}")))
}

/// A browser download: a Blob behind a transient `<a download>` that is clicked.
````

In `linkage-sim-rs/src/gui/export/download.rs`, replace:

````rust
    else {
        return DownloadOutcome::Cancelled;
    };
    match std::fs::write(&path, contents) {
        Ok(()) => DownloadOutcome::Saved(format!("Saved: {}", path.display())),
        Err(e) => DownloadOutcome::Failed(format!("Write failed: {}", e)),
    }
````

with:

````rust
    else {
        return DownloadOutcome::Cancelled;
    };
    write_file(&path, contents)
}

/// Writes `contents` to `path` byte for byte (the native save after its dialog).
#[cfg(feature = "native")]
fn write_file(path: &std::path::Path, contents: &[u8]) -> DownloadOutcome {
    match std::fs::write(path, contents) {
        Ok(()) => DownloadOutcome::Saved(format!("Saved: {}", path.display())),
        Err(e) => DownloadOutcome::Failed(format!("Write failed: {}", e)),
    }
````

In `linkage-sim-rs/src/gui/calculator_window.rs`, replace:

````rust
                    mime,
                    contents,
                } => {
                    let outcome = download::download_text(
                        &file_name,
                        mime,
                        &contents,
````

with:

````rust
                    mime,
                    contents,
                } => {
                    let outcome = download::download_bytes(
                        &file_name,
                        mime,
                        &contents,
````


- [ ] **Step 4: Run the tests to see them pass**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-xlsx/magcoupling-rs/Cargo.toml --features app --lib -- files:: the_export_buttons save_design_queues json_export_in_torque 2>&1 | grep -E "^test |test result"
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-xlsx/linkage-sim-rs/Cargo.toml --lib -- download:: calculator_window 2>&1 | grep -E "test result"
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-xlsx/magcoupling-rs/Cargo.toml --features app --lib 2>&1 | grep -E "test result"
```

Expected: the four tests `ok` (`a_saved_file_holds_the_bytes_as_they_are`, `save_design_queues_the_design_file_and_load_design_asks_for_one`, `the_json_export_in_torque_to_magnets_holds_the_inputs_that_produced_the_results`, `the_export_buttons_queue_the_files_for_the_host`) and `test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 494 filtered out`; linkage `test result: ok. 26 passed; 0 failed; 0 ignored; 0 measured; 930 filtered out` (the new download test and the calculator window's 25); the magcoupling library `test result: ok. 498 passed`.

- [ ] **Step 5: Format, lint and gate**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-xlsx/magcoupling-rs/Cargo.toml --check
cargo clippy --manifest-path C:/Users/Cole/source/repos/lsim-mag-xlsx/magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings 2>&1 | tail -n 1
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-xlsx/linkage-sim-rs/scripts/gate.sh 2>&1 | tail -n 3
git -C C:/Users/Cole/source/repos/lsim-mag-xlsx checkout -- docs/chebyshev_lambda
```

Expected: `fmt --check` prints nothing; clippy ends `Finished`; the gate ends `GATE PASS` with no `SKIP gate` line.

- [ ] **Step 6: Commit**

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-xlsx add magcoupling-rs/src/gui/panel.rs magcoupling-rs/src/app/files.rs linkage-sim-rs/src/gui/export/download.rs linkage-sim-rs/src/gui/calculator_window.rs
git -C C:/Users/Cole/source/repos/lsim-mag-xlsx commit -m "feat(magcoupling-rs): SaveFile carries bytes, both hosts write them (decision X-10)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9"
git -C C:/Users/Cole/source/repos/lsim-mag-xlsx status --short
```

Expected: one commit; no status lines.

### Task 2: The spreadsheet's layout (`gui::spreadsheet`)

**Model:** `sonnet` (exact code; CLAUDE.md section 5). Escalate a retry to the session model.

**Files:**
- Create: `magcoupling-rs/src/gui/spreadsheet.rs` (tests first, then the module above them)
- Modify: `magcoupling-rs/src/gui/mod.rs` (`pub mod spreadsheet;`)
- Modify: `magcoupling-rs/src/gui/corrections.rs` (`marks_values` made `pub(crate)`)
- Modify: `magcoupling-rs/src/gui/inputs.rs` (`InputGroup::plain_sections`, `advanced_sections`, `has_advanced`; `InputSection::heading_in`, which `SectionMatches::heading` now calls; a test)
- Modify: `magcoupling-rs/src/gui/results_table.rs` (`RUST_ONLY` for the row's cell column; `entry_level`, which `ResultsTable::ui`'s rows now call, and `worst_level` through it, which its headings now call; a test)
- Modify: `magcoupling-rs/src/gui/panel.rs` (`assumptions_banner`, which `header_ui` now calls; `inputs_ui` and `section_ui` draw the sections with the inputs module's helpers)

**Interfaces:**
- Consumes: `InputCatalogue::get().groups_in(InputOrder) -> &[InputGroup]` (each `InputGroup { name, label, sections: Vec<InputSection { id, label, advanced, entries: Vec<InputEntry { path, meta, default, .. }> }> }`), `KEY_DESIGN`, `ADVANCED_HEADING` (inputs); `result_groups() -> &'static [ResultGroup { id, label, other, rows }]`, `OTHER_RESULTS` (result_groups); `table_entries() -> &'static [TableEntry { path, info, marker, check, .. }]`, `exact_number`, `table_lines(matches: &[usize], order: ResultOrder, failing: Option<&[usize]>, open: &dyn Fn(usize) -> bool) -> Vec<Line>` with `Line { OtherResults, Group { group, shown, open }, Row(usize) }` and `ResultOrder::Grouped` (results_table); `dashboard_lines`, `check_level` (through `entry_level`; the tests call it directly), `warning_lines`, `severity_level`, `end_effect_banner`, `result_info`, `Level`, `STORED_3D_LABEL`, `WARNINGS_HEADING` (dashboard); `registry().equation_for(path)` and `engine::explain::render::plain(registry, &eq.symbol, &eq.formula)`; `assumptions::states(&DesignInputs)`; `REGISTRY` (deviations); `HEADING`, `SIZED_NOTE`, `TARGET_LABEL`, `FREE_VARIABLE_LABEL` (panel); `BLANK_TEXT` (input_ui); `SizingState`, `SizingMode`, `variable_label`, `TARGET_RANGE_INPUT` (sizing).
- Produces: `pub struct Snapshot<'a> { inputs: &'a DesignInputs, results: &'a DesignResults, sizing: SizingState, sizing_status: Option<String>, input_order: InputOrder, share_link: String, exported_unix_s: i64 }`; `pub fn sheets(snapshot: &Snapshot) -> Vec<Sheet>` (four sheets, in order); `pub struct Sheet { name: &'static str, widths: &'static [f64], frozen_rows: u32, autofilter: bool, rows: Vec<Vec<Cell>> }` (the Summary `0, false`, the other three `1, true`); `pub struct Cell { value: CellValue, style: CellStyle }` with `Cell::empty()`, `Cell::number(f64)`, `Cell::text(impl Into<String>)` (an empty text is `CellValue::Empty`), `styled`, `as_text`; `pub enum CellValue { Empty, Number(f64), Text(String), Time(i64) }`; `pub enum CellStyle { Plain, Title, Header, Group, Section, Level(Level) }`; `pub fn value_cell(&Value) -> Cell`; `pub fn level_cell(Option<Level>) -> Cell`; the consts `SUMMARY_SHEET`, `INPUTS_SHEET`, `RESULTS_SHEET`, `ASSUMPTIONS_SHEET`, `INPUT_COLUMNS`, `RESULT_COLUMNS`, `DASHBOARD_COLUMNS`, `ASSUMPTION_COLUMNS`, `EXPORTED`, `APP`, `SHARE_LINK`, `SIZING_MODE`, `SIZING_OUTCOME`, `CORRECTIONS_HEADING`, `NO_WARNING`, `GREYED_NOTE`, `YES`; `results_table::RUST_ONLY`, `results_table::entry_level(&DesignResults, &TableEntry) -> Option<Level>`, `results_table::worst_level(&DesignResults, &[usize]) -> Option<Level>`; `InputGroup::plain_sections(&self) -> impl Iterator<Item = &InputSection>`, `InputGroup::advanced_sections(&self) -> impl Iterator<Item = &InputSection>`, `InputGroup::has_advanced(&self) -> bool`, `InputSection::heading_in(&self, group: &InputGroup) -> Option<&'static str>`; `panel::assumptions_banner(&DesignInputs) -> Option<String>` (`pub(crate)`); `corrections::marks_values` (`pub(crate)`).

- [ ] **Step 1: Write the failing tests**

Create `magcoupling-rs/src/gui/spreadsheet.rs`:

````rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_all;
    use crate::engine::assumptions::ASSUMPTIONS;
    use crate::engine::sizing::FreeVariable;
    use crate::gui::dashboard::{CHECKS, DASHBOARD, check_level};
    use crate::gui::test_support::short_magnets;

    /// The snapshot of `inputs` and `results` in Magnets -> Torque, the workflow order.
    fn snapshot<'a>(inputs: &'a DesignInputs, results: &'a DesignResults) -> Snapshot<'a> {
        Snapshot {
            inputs,
            results,
            sizing: SizingState::default(),
            sizing_status: None,
            input_order: InputOrder::Workflow,
            share_link: "https://example.test/magcoupling/?m=abc".to_owned(),
            exported_unix_s: 1_790_000_000,
        }
    }

    fn sheet<'s>(book: &'s [Sheet], name: &str) -> &'s Sheet {
        book.iter()
            .find(|sheet| sheet.name == name)
            .unwrap_or_else(|| panic!("no sheet {name}"))
    }

    /// The index of the column headed `name`.
    fn column(columns: &[&str], name: &str) -> usize {
        columns
            .iter()
            .position(|column| *column == name)
            .unwrap_or_else(|| panic!("no column {name}"))
    }

    /// The rows below the header that name a path in column `path`: one per input or result.
    fn path_rows(sheet: &Sheet, path: usize) -> Vec<&Vec<Cell>> {
        sheet.rows[sheet.frozen_rows as usize..]
            .iter()
            .filter(|row| row.len() > path && !row[path].as_text().is_empty())
            .collect()
    }

    /// The row of `sheet` whose column `path` holds `wanted`.
    fn row_of<'s>(sheet: &'s Sheet, path: usize, wanted: &str) -> &'s Vec<Cell> {
        path_rows(sheet, path)
            .into_iter()
            .find(|row| row[path].as_text() == wanted)
            .unwrap_or_else(|| panic!("no row {wanted}"))
    }

    /// The Summary row whose first cell reads `label`.
    fn summary_row<'s>(book: &'s [Sheet], label: &str) -> Option<&'s Vec<Cell>> {
        sheet(book, SUMMARY_SHEET)
            .rows
            .iter()
            .find(|row| row.first().map(Cell::as_text) == Some(label))
    }

    #[test]
    fn the_workbook_has_the_summary_inputs_results_and_assumptions_sheets() {
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        let book = sheets(&snapshot(&inputs, &results));
        let names: Vec<&str> = book.iter().map(|sheet| sheet.name).collect();
        assert_eq!(
            names,
            [
                SUMMARY_SHEET,
                INPUTS_SHEET,
                RESULTS_SHEET,
                ASSUMPTIONS_SHEET
            ]
        );
        for (sheet, columns) in [
            (&book[1], &INPUT_COLUMNS[..]),
            (&book[2], &RESULT_COLUMNS[..]),
            (&book[3], &ASSUMPTION_COLUMNS[..]),
        ] {
            assert_eq!(sheet.frozen_rows, 1, "{}", sheet.name);
            let header: Vec<&str> = sheet.rows[0].iter().map(Cell::as_text).collect();
            assert_eq!(header, columns, "{}", sheet.name);
            assert!(
                sheet.rows[0].iter().all(|c| c.style == CellStyle::Header),
                "{}",
                sheet.name
            );
        }
        // The Summary is read top to bottom: nothing frozen, no filter; each table sheet's
        // header row is frozen and filters its columns.
        let frozen: Vec<(u32, bool)> = book.iter().map(|s| (s.frozen_rows, s.autofilter)).collect();
        assert_eq!(frozen, [(0, false), (1, true), (1, true), (1, true)]);
        assert_eq!(
            book[0].rows[0],
            [Cell::text(HEADING).styled(CellStyle::Title)]
        );
        for sheet in &book {
            assert_eq!(
                sheet.widths.len(),
                sheet.rows.iter().map(Vec::len).max().unwrap()
            );
        }
    }

    #[test]
    fn every_input_is_on_one_row_with_its_value_unit_and_cell_in_either_order() {
        let mut inputs = DesignInputs::default();
        inputs.metal.face_gap_mm = 1.41;
        let results = compute_all(&inputs);
        let catalogue = InputCatalogue::get();
        let path = column(&INPUT_COLUMNS, "Path");
        for order in InputOrder::ALL {
            let mut shot = snapshot(&inputs, &results);
            shot.input_order = order;
            let book = sheets(&shot);
            let rows = path_rows(sheet(&book, INPUTS_SHEET), path);
            for row in &rows {
                let entry = catalogue.entry(row[path].as_text()).expect("an input");
                assert_eq!(row[0], Cell::text(entry.meta.label));
                assert_eq!(row[1], value_cell(&inputs.get(&entry.path).unwrap()));
                assert_eq!(row[2], Cell::text(entry.meta.unit));
                assert_eq!(row[3], Cell::text(entry.meta.cell.unwrap_or(RUST_ONLY)));
            }
            let mut paths: Vec<&str> = rows.iter().map(|row| row[path].as_text()).collect();
            assert_eq!(paths.len(), catalogue.all().count(), "{order:?}");
            paths.sort_unstable();
            paths.dedup();
            assert_eq!(
                paths.len(),
                catalogue.all().count(),
                "each input once ({order:?})"
            );
        }
    }

    #[test]
    fn the_inputs_follow_the_order_shown_under_their_group_section_and_advanced_headings() {
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        let catalogue = InputCatalogue::get();
        let path = column(&INPUT_COLUMNS, "Path");
        let advanced = column(&INPUT_COLUMNS, "Advanced");
        for order in InputOrder::ALL {
            let mut shot = snapshot(&inputs, &results);
            shot.input_order = order;
            let book = sheets(&shot);
            let sheet = sheet(&book, INPUTS_SHEET);
            let groups = catalogue.groups_in(order);
            let headings: Vec<&str> = sheet
                .rows
                .iter()
                .filter(|row| row[0].style == CellStyle::Group)
                .map(|row| row[0].as_text())
                .collect();
            let want: Vec<&str> = groups.iter().map(|group| group.label).collect();
            assert_eq!(headings, want, "{order:?}");
            // The inputs in exactly the order the page draws them: group by group, the plain
            // sections, then the advanced ones, each section's inputs in order.
            let got: Vec<&str> = path_rows(sheet, path)
                .iter()
                .map(|row| row[path].as_text())
                .collect();
            let drawn: Vec<&str> = groups
                .iter()
                .flat_map(|g| g.plain_sections().chain(g.advanced_sections()))
                .flat_map(|s| s.entries.iter().map(|e| e.path.as_str()))
                .collect();
            assert_eq!(got, drawn, "{order:?}");
            // Each input under its own group, after its section's heading, and an advanced input
            // only after its group's Advanced heading.
            let mut group: Option<&InputGroup> = None;
            let mut section: Option<&str> = None;
            let mut in_advanced = false;
            let mut advanced_headings = 0;
            for row in &sheet.rows[1..] {
                match row[0].style {
                    CellStyle::Group => {
                        group = groups.iter().find(|g| g.label == row[0].as_text());
                        section = None;
                        in_advanced = false;
                    }
                    CellStyle::Section if row[0].as_text() == ADVANCED_HEADING => {
                        in_advanced = true;
                        advanced_headings += 1;
                    }
                    CellStyle::Section => section = Some(row[0].as_text()),
                    _ => {
                        let input = row[path].as_text();
                        let (holder, held) = catalogue.section_of(order, input).expect(input);
                        assert_eq!(group.map(|g| &g.name), Some(&holder.name), "{input}");
                        assert_eq!(held.advanced, in_advanced, "{input}");
                        assert_eq!(row[advanced], flag(held.advanced), "{input}");
                        if held.id != holder.name {
                            assert_eq!(section, Some(held.label), "{input}");
                        }
                    }
                }
            }
            let want_advanced = groups
                .iter()
                .filter(|g| g.sections.iter().any(|s| s.advanced))
                .count();
            assert_eq!(advanced_headings, want_advanced, "{order:?}");
            // A group's own section has no heading (`InputSection`'s rule): no section heading
            // repeats its group's, and there is one per other section, plus the Advanced ones.
            let mut current = "";
            let mut section_headings = 0;
            for row in &sheet.rows[1..] {
                match row[0].style {
                    CellStyle::Group => current = row[0].as_text(),
                    CellStyle::Section => {
                        assert_ne!(row[0].as_text(), current, "{order:?}");
                        section_headings += 1;
                    }
                    _ => {}
                }
            }
            let other_sections = groups
                .iter()
                .flat_map(|g| g.sections.iter().filter(|s| s.id != g.name))
                .count();
            assert_eq!(
                section_headings,
                other_sections + want_advanced,
                "{order:?}"
            );
        }
        // The workbook order has no Advanced heading; the workflow order has some.
        let count = |order: InputOrder| {
            let mut shot = snapshot(&inputs, &results);
            shot.input_order = order;
            sheets(&shot)[1]
                .rows
                .iter()
                .filter(|row| row[0].as_text() == ADVANCED_HEADING)
                .count()
        };
        assert_eq!(count(InputOrder::Workbook), 0);
        assert!(count(InputOrder::Workflow) > 0);
    }

    #[test]
    fn an_input_row_flags_changes_assumptions_key_design_and_notes_choices_and_blanks() {
        let mut inputs = DesignInputs::default();
        inputs.metal.face_gap_mm = 1.41;
        let results = compute_all(&inputs);
        let book = sheets(&snapshot(&inputs, &results));
        let sheet = sheet(&book, INPUTS_SHEET);
        let path = column(&INPUT_COLUMNS, "Path");
        let at = |name: &str| column(&INPUT_COLUMNS, name);
        let row = |input: &str| row_of(sheet, path, input);
        let face_gap = row("metal.face_gap_mm");
        assert_eq!(face_gap[1], Cell::number(1.41));
        assert_eq!(face_gap[at("Changed from default")], Cell::text(YES));
        assert_eq!(face_gap[at("Key design")], Cell::text(YES));
        assert_eq!(face_gap[at("Assumption")], Cell::empty());
        assert_eq!(face_gap[at("Note")], Cell::empty());
        // An assumption at its default.
        let variation = row("metal.variation");
        assert_eq!(variation[at("Assumption")], Cell::text(YES));
        assert_eq!(variation[at("Changed from default")], Cell::empty());
        assert_eq!(variation[at("Key design")], Cell::empty());
        // A selector: its code a number, its choice in the note.
        let entry = InputCatalogue::get().entry("coupling.backiron").unwrap();
        let Some(Value::Int(code)) = inputs.get("coupling.backiron") else {
            panic!("a selector code")
        };
        let (_, choice) = entry.meta.choices.iter().find(|(c, _)| *c == code).unwrap();
        let backiron = row("coupling.backiron");
        assert_eq!(backiron[1], Cell::number(code as f64));
        assert_eq!(
            backiron[at("Note")],
            Cell::text(format!("{code} = {choice}"))
        );
        // An optional input left blank: no value, noted.
        let drag = row("metal.measured_drag_Nm");
        assert_eq!(drag[1], Cell::empty());
        assert_eq!(drag[at("Note")], Cell::text(BLANK_TEXT));
        // A text input: the part name as it is, with no note.
        let part = row("coupling.magnets.part_inner");
        assert_eq!(
            part[1],
            Cell::text(inputs.coupling.magnets.part_inner.clone())
        );
        assert_eq!(part[at("Note")], Cell::empty());
        // A Rust-only input names no cell; an advanced input is flagged.
        let catalogue = InputCatalogue::get();
        let rust_only = catalogue.all().find(|e| e.meta.cell.is_none()).unwrap();
        assert_eq!(
            row(&rust_only.path)[at("Workbook cell")],
            Cell::text(RUST_ONLY)
        );
        let advanced = catalogue
            .workflow
            .iter()
            .flat_map(|g| g.sections.iter())
            .find(|s| s.advanced)
            .unwrap();
        assert_eq!(
            row(&advanced.entries[0].path)[at("Advanced")],
            Cell::text(YES)
        );
        // An empty text input (manual magnets: no part name): no value, noted blank.
        let mut manual = inputs.clone();
        manual.coupling.magnets.part_inner.clear();
        let results = compute_all(&manual);
        let book = sheets(&snapshot(&manual, &results));
        let manual_inputs = &book[1];
        assert_eq!(manual_inputs.name, INPUTS_SHEET);
        let part = row_of(manual_inputs, path, "coupling.magnets.part_inner");
        assert_eq!(part[1], Cell::empty());
        assert_eq!(part[at("Note")], Cell::text(BLANK_TEXT));
    }

    #[test]
    fn every_result_is_on_one_row_by_physics_chain_with_its_value() {
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        let book = sheets(&snapshot(&inputs, &results));
        let sheet = sheet(&book, RESULTS_SHEET);
        let path = column(&RESULT_COLUMNS, "Path");
        let rows = path_rows(sheet, path);
        let entries = table_entries();
        // The results table's grouped lines, every row matched and every group open.
        let all: Vec<usize> = (0..entries.len()).collect();
        let lines = table_lines(&all, ResultOrder::Grouped, None, &|_| true);
        // Exactly the table's rows, each once, in its grouped order.
        let want: Vec<&str> = lines
            .iter()
            .filter_map(|line| match line {
                Line::Row(index) => Some(entries[*index].path.as_str()),
                _ => None,
            })
            .collect();
        let got: Vec<&str> = rows.iter().map(|row| row[path].as_text()).collect();
        assert_eq!(got, want);
        assert_eq!(got.len(), entries.len());
        for row in &rows {
            let result = row[path].as_text();
            let info = result_info(result).unwrap();
            assert_eq!(row[0], Cell::text(info.meta.label), "{result}");
            assert_eq!(
                row[1],
                value_cell(&results.get(result).unwrap()),
                "{result}"
            );
            assert_eq!(row[2], Cell::text(info.meta.unit), "{result}");
            assert_eq!(
                row[4],
                Cell::text(info.cell.as_deref().unwrap_or(RUST_ONLY)),
                "{result}"
            );
        }
        // The headings: the table's, in its order, the package groups under "Other results"
        // (once, a group heading) as headings inside it.
        let headings: Vec<(&str, CellStyle)> = sheet
            .rows
            .iter()
            .filter(|row| matches!(row[0].style, CellStyle::Group | CellStyle::Section))
            .map(|row| (row[0].as_text(), row[0].style))
            .collect();
        let want: Vec<(&str, CellStyle)> = lines
            .iter()
            .filter_map(|line| match line {
                Line::OtherResults => Some((OTHER_RESULTS, CellStyle::Group)),
                Line::Group { group, .. } => {
                    let group = &result_groups()[*group];
                    let style = if group.other {
                        CellStyle::Section
                    } else {
                        CellStyle::Group
                    };
                    Some((group.label, style))
                }
                Line::Row(_) => None,
            })
            .collect();
        assert_eq!(headings, want);
        assert_eq!(
            headings[0].0,
            result_groups()[0].label,
            "the headline first"
        );
        let other = headings.iter().filter(|(text, _)| *text == OTHER_RESULTS);
        assert_eq!(other.count(), 1);
        // The pull-out: its number at full precision and its corrections.
        let pullout = row_of(sheet, path, "model.pullout_Nm");
        assert_eq!(pullout[1], Cell::number(results.model.pullout_Nm));
        assert_eq!(
            pullout[column(&RESULT_COLUMNS, "Corrected vs workbook")],
            Cell::text("E3 E7 E8")
        );
    }

    #[test]
    fn a_check_carries_its_level_and_a_greyed_check_none() {
        let check = column(&RESULT_COLUMNS, "Check");
        let path = column(&RESULT_COLUMNS, "Path");
        // The default design, one with six failing checks (manual magnets) and one out of the
        // end-effect range (short magnets).
        let mut manual = DesignInputs::default();
        manual.coupling.magnets.part_inner.clear();
        manual.coupling.magnets.part_outer.clear();
        for inputs in [DesignInputs::default(), manual.clone(), short_magnets()] {
            let results = compute_all(&inputs);
            let book = sheets(&snapshot(&inputs, &results));
            let sheet = sheet(&book, RESULTS_SHEET);
            for row in path_rows(sheet, path) {
                let result = row[path].as_text();
                let want = if CHECKS.contains(&result) {
                    level_cell(check_level(&results, result))
                } else {
                    Cell::empty()
                };
                assert_eq!(row[check], want, "{result}");
            }
            // Each group heading shows the worst level of its checks.
            for group in result_groups() {
                let heading = sheet
                    .rows
                    .iter()
                    .find(|row| row[0].as_text() == group.label && row.len() > check)
                    .unwrap();
                assert_eq!(
                    heading[check],
                    level_cell(worst_level(&results, &group.rows))
                );
            }
        }
        // Manual magnets: red and amber cells.
        let results = compute_all(&manual);
        let book = sheets(&snapshot(&manual, &results));
        let styles: Vec<CellStyle> = path_rows(sheet(&book, RESULTS_SHEET), path)
            .iter()
            .map(|row| row[check].style)
            .collect();
        assert!(styles.contains(&CellStyle::Level(Level::Bad)));
        assert!(styles.contains(&CellStyle::Level(Level::Caution)));
        // Short magnets: the hot minimum (red by its text) is greyed, so it has no level; the
        // Summary carries the end-effect banner in red and notes the greyed dashboard rows.
        let inputs = short_magnets();
        let results = compute_all(&inputs);
        let book = sheets(&snapshot(&inputs, &results));
        let hot_min = row_of(sheet(&book, RESULTS_SHEET), path, "metal.hot_min_check");
        assert_eq!(hot_min[check], Cell::empty());
        let banner = end_effect_banner(&results).expect("out of range");
        let summary = &sheet(&book, SUMMARY_SHEET).rows;
        assert!(summary.contains(&vec![
            Cell::text(banner).styled(CellStyle::Level(Level::Bad))
        ]));
        let pullout = summary
            .iter()
            .find(|row| row.get(5).map(Cell::as_text) == Some("model.pullout_Nm"))
            .unwrap();
        assert_eq!(pullout[7], Cell::text(GREYED_NOTE));
        assert_eq!(pullout[3], Cell::empty());
    }

    #[test]
    fn a_number_that_is_not_finite_is_the_csv_s_text() {
        assert_eq!(value_cell(&Value::Num(f64::INFINITY)), Cell::text("+inf"));
        assert_eq!(
            value_cell(&Value::Num(f64::NEG_INFINITY)),
            Cell::text("-inf")
        );
        assert_eq!(value_cell(&Value::Num(f64::NAN)), Cell::text("NaN"));
        assert_eq!(value_cell(&Value::Int(10)), Cell::number(10.0));
        assert_eq!(value_cell(&Value::Text(String::new())), Cell::empty());
        assert_eq!(value_cell(&Value::None), Cell::empty());
        // A positive beta typed in for manual magnets with no grade (E20): no hot limit, +inf,
        // and no torque at it, NaN.
        let mut inputs = DesignInputs::default();
        inputs.coupling.magnets.part_inner = String::new();
        inputs.coupling.magnets.part_outer = String::new();
        inputs.temperature.demag.coercivity_source = 0;
        inputs.temperature.demag.beta_hcj_per_C = 0.0035;
        let results = compute_all(&inputs);
        let book = sheets(&snapshot(&inputs, &results));
        let sheet = sheet(&book, RESULTS_SHEET);
        let path = column(&RESULT_COLUMNS, "Path");
        assert_eq!(
            row_of(sheet, path, "temperature.demag.magnet_limit_C")[1],
            Cell::text("+inf")
        );
        assert_eq!(
            row_of(sheet, path, "temperature.demag.torque_at_limit_Nm")[1],
            Cell::text("NaN")
        );
        // No number cell on any sheet holds a number that is not finite.
        for sheet in &book {
            for cell in sheet.rows.iter().flatten() {
                if let CellValue::Number(x) = cell.value {
                    assert!(x.is_finite(), "{}: {x}", sheet.name);
                }
            }
        }
    }

    #[test]
    fn the_equation_column_holds_the_record_s_plain_formula() {
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        let book = sheets(&snapshot(&inputs, &results));
        let sheet = sheet(&book, RESULTS_SHEET);
        let path = column(&RESULT_COLUMNS, "Path");
        let equation = column(&RESULT_COLUMNS, "Equation");
        assert_eq!(
            row_of(sheet, path, "model.f_end")[equation],
            Cell::text("f_{end} = 1 − c_{end} · (τ_p/L)")
        );
        // Exactly the results with an equation record have one (the space claim has none,
        // decision G2).
        let rows = path_rows(sheet, path);
        for row in &rows {
            let result = row[path].as_text();
            let explained = registry().equation_for(result).is_some();
            assert_eq!(!row[equation].as_text().is_empty(), explained, "{result}");
        }
        assert_eq!(
            row_of(sheet, path, "housing.space_claim_check")[equation],
            Cell::empty()
        );
        let explained = rows
            .iter()
            .filter(|r| !r[equation].as_text().is_empty())
            .count();
        assert!(explained > 300, "{explained}");
    }

    #[test]
    fn the_summary_holds_the_export_the_link_the_dashboard_the_warnings_and_the_corrections() {
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        let shot = snapshot(&inputs, &results);
        let book = sheets(&shot);
        let row = |label: &str| summary_row(&book, label).unwrap_or_else(|| panic!("{label}"));
        assert_eq!(row(EXPORTED)[1].value, CellValue::Time(1_790_000_000));
        assert_eq!(
            row(APP)[1],
            Cell::text(format!("magcoupling-rs {}", env!("CARGO_PKG_VERSION")))
        );
        assert_eq!(row(SHARE_LINK)[1], Cell::text(shot.share_link.clone()));
        assert_eq!(
            row(SIZING_MODE)[1],
            Cell::text(SizingMode::MagnetsToTorque.label())
        );
        assert!(summary_row(&book, FREE_VARIABLE_LABEL).is_none());
        // No banner at the default design.
        let rows = &sheet(&book, SUMMARY_SHEET).rows;
        assert!(
            !rows
                .iter()
                .flatten()
                .any(|c| matches!(c.style, CellStyle::Level(_))
                    && c.as_text().starts_with("Assumptions modified"))
        );
        // The dashboard: its header, then each of its rows with value, unit, badge and path.
        let header = rows
            .iter()
            .position(|r| r.first().map(Cell::as_text) == Some(DASHBOARD_COLUMNS[0]))
            .unwrap();
        let lines = dashboard_lines(&results);
        assert_eq!(lines.len(), DASHBOARD.len());
        for (row, line) in rows[header + 1..].iter().zip(&lines) {
            assert_eq!(row[0], Cell::text(line.label));
            assert_eq!(row[1], value_cell(&results.get(line.path).unwrap()));
            assert_eq!(row[3], level_cell(line.level));
            assert_eq!(row[5], Cell::text(line.path));
            assert_eq!(row[6], Cell::text(line.marker.clone()));
        }
        assert_eq!(rows[header + 1 + lines.len()], Vec::<Cell>::new());
        // No warning fires at the default design.
        assert!(summary_row(&book, WARNINGS_HEADING).is_some());
        assert!(summary_row(&book, NO_WARNING).is_some());
        // The corrections applied, one row each, last.
        let corrections = rows
            .iter()
            .position(|r| r.first().map(Cell::as_text) == Some(CORRECTIONS_HEADING))
            .unwrap();
        let applied: Vec<_> = REGISTRY.iter().filter(|d| marks_values(d)).collect();
        assert!(applied.len() >= 19, "{}", applied.len());
        assert_eq!(
            rows.len(),
            corrections + 1 + applied.len(),
            "the corrections are last"
        );
        for (row, deviation) in rows[corrections + 1..].iter().zip(&applied) {
            assert_eq!(row[0], Cell::text(deviation.id.to_string()));
            assert_eq!(row[1], Cell::text(deviation.title));
        }
    }

    #[test]
    fn the_summary_lists_the_warnings_that_fire_and_the_assumptions_banner() {
        // 304 stainless back iron: a warning (red) and a caution (amber) fire.
        let mut inputs = DesignInputs::default();
        inputs.materials.parts.back_iron = 7;
        inputs.metal.variation = 0.2;
        let results = compute_all(&inputs);
        let book = sheets(&snapshot(&inputs, &results));
        let rows = &sheet(&book, SUMMARY_SHEET).rows;
        let heading = rows
            .iter()
            .position(|r| r.first().map(Cell::as_text) == Some(WARNINGS_HEADING))
            .unwrap();
        let lines = warning_lines(&results);
        assert_eq!(lines.len(), 2);
        for (row, (rule, text)) in rows[heading + 1..].iter().zip(&lines) {
            assert_eq!(
                row[0].style,
                CellStyle::Level(severity_level(rule.severity))
            );
            assert_eq!(row[1], Cell::text(text.clone()));
        }
        assert_eq!(rows[heading + 1][0].as_text(), "Warning");
        assert!(summary_row(&book, NO_WARNING).is_none());
        // The assumptions banner, amber, as the header shows it.
        let banner = assumptions_banner(&inputs).expect("the variation is an assumption");
        assert!(rows.contains(&vec![
            Cell::text(banner).styled(CellStyle::Level(Level::Caution))
        ]));
    }

    #[test]
    fn torque_to_magnets_shows_the_sizing_and_notes_the_free_variable_s_row() {
        // The design shown: the solved axial length in the override.
        let mut inputs = DesignInputs::default();
        inputs.coupling.magnets.axial_length_mm = Some(14.2);
        let results = compute_all(&inputs);
        let mut shot = snapshot(&inputs, &results);
        shot.sizing = SizingState {
            mode: SizingMode::TorqueToMagnets,
            variable: FreeVariable::AxialLength,
            target_Nm: 2.5,
        };
        shot.sizing_status = Some("Solved: 14.2 mm".to_owned());
        let book = sheets(&shot);
        let row = |label: &str| summary_row(&book, label).unwrap_or_else(|| panic!("{label}"));
        assert_eq!(
            row(SIZING_MODE)[1],
            Cell::text(SizingMode::TorqueToMagnets.label())
        );
        assert_eq!(
            row(FREE_VARIABLE_LABEL)[1],
            Cell::text(variable_label(FreeVariable::AxialLength))
        );
        assert_eq!(row(TARGET_LABEL)[1], Cell::number(2.5));
        assert_eq!(row(TARGET_LABEL)[2], Cell::text("N·m"));
        assert_eq!(row(SIZING_OUTCOME)[1], Cell::text("Solved: 14.2 mm"));
        // The free variable's row: the value shown, noted; no other row carries the note.
        let sheet = sheet(&book, INPUTS_SHEET);
        let path = column(&INPUT_COLUMNS, "Path");
        let note = column(&INPUT_COLUMNS, "Note");
        let length = row_of(sheet, path, FreeVariable::AxialLength.path());
        assert_eq!(length[1], Cell::number(14.2));
        assert_eq!(length[note], Cell::text(SIZED_NOTE));
        let noted = path_rows(sheet, path)
            .iter()
            .filter(|r| r[note].as_text().contains(SIZED_NOTE))
            .count();
        assert_eq!(noted, 1);
    }

    #[test]
    fn the_assumptions_sheet_lists_each_assumption_s_inputs_with_their_workbook_defaults() {
        let mut inputs = DesignInputs::default();
        inputs.metal.variation = 0.2;
        let results = compute_all(&inputs);
        let book = sheets(&snapshot(&inputs, &results));
        let sheet = sheet(&book, ASSUMPTIONS_SHEET);
        let defaults = DesignInputs::default();
        let rows = &sheet.rows[1..];
        let want: Vec<(&str, &str)> = ASSUMPTIONS
            .iter()
            .flat_map(|a| a.paths.iter().map(move |p| (a.label, *p)))
            .collect();
        assert_eq!(rows.len(), want.len());
        for (row, (label, path)) in rows.iter().zip(want) {
            assert_eq!(row[0], Cell::text(label));
            assert_eq!(row[1], Cell::text(path));
            assert_eq!(row[2], value_cell(&inputs.get(path).unwrap()), "{path}");
            assert_eq!(row[4], value_cell(&defaults.get(path).unwrap()), "{path}");
            let changed = path == "metal.variation";
            assert_eq!(row[5], flag(changed), "{path}");
        }
        assert!(
            rows.iter().all(|row| !row[6].as_text().is_empty()),
            "every rationale"
        );
    }
}
````

In `magcoupling-rs/src/gui/mod.rs`, replace:

````rust
pub mod results_table;
pub mod session;
pub mod sizing;
#[cfg(test)]
pub(crate) mod test_support;
pub mod trace;
````

with:

````rust
pub mod results_table;
pub mod session;
pub mod sizing;
pub mod spreadsheet;
#[cfg(test)]
pub(crate) mod test_support;
pub mod trace;
````

In `magcoupling-rs/src/gui/inputs.rs`, replace:

````rust
    }

    #[test]
    fn the_filter_matches_label_path_and_cell_by_section_in_the_order_shown() {
        let catalogue = InputCatalogue::new();
        let paths = |order: InputOrder, query: &str| -> Vec<String> {
````

with:

````rust
    }

    #[test]
    fn a_group_lists_its_sections_and_their_headings_as_the_page_draws_them() {
        let catalogue = InputCatalogue::new();
        let group = |order: InputOrder, name: &str| -> &InputGroup {
            catalogue
                .groups_in(order)
                .iter()
                .find(|group| group.name == name)
                .unwrap_or_else(|| panic!("no group {name}"))
        };
        let ids = |sections: Vec<&InputSection>| -> Vec<String> {
            sections.iter().map(|section| section.id.clone()).collect()
        };
        // The workflow's gap group: its own section and the allowances in view, the bedding
        // clearances under its Advanced heading; its own section has no heading.
        let gap = group(InputOrder::Workflow, "gap");
        assert_eq!(
            ids(gap.plain_sections().collect()),
            ["gap", "gap.allowances"]
        );
        assert_eq!(ids(gap.advanced_sections().collect()), ["gap.bedding"]);
        assert!(gap.has_advanced());
        let headings: Vec<Option<&str>> = gap.sections.iter().map(|s| s.heading_in(gap)).collect();
        assert_eq!(
            headings,
            [
                None,
                Some("Running-clearance allowances"),
                Some("Bedding clearances")
            ]
        );
        // A workflow group without an advanced section, and a package group: the group's own
        // fields have no heading, a nested group its label.
        let magnets = group(InputOrder::Workflow, "magnets");
        assert!(!magnets.has_advanced());
        assert_eq!(magnets.advanced_sections().count(), 0);
        let coupling = group(InputOrder::Workbook, "coupling");
        assert_eq!(coupling.sections[0].heading_in(coupling), None);
        assert_eq!(coupling.sections[1].heading_in(coupling), Some("Magnets"));
        // In either order every section once, the plain ones first: the order drawn.
        for order in InputOrder::ALL {
            for group in catalogue.groups_in(order) {
                let drawn: Vec<&InputSection> = group
                    .plain_sections()
                    .chain(group.advanced_sections())
                    .collect();
                let all: Vec<&InputSection> = group.sections.iter().collect();
                assert_eq!(drawn, all, "{}", group.name);
            }
        }
        // The workbook order has no Advanced heading.
        assert!(!catalogue.groups.iter().any(InputGroup::has_advanced));
    }

    #[test]
    fn the_filter_matches_label_path_and_cell_by_section_in_the_order_shown() {
        let catalogue = InputCatalogue::new();
        let paths = |order: InputOrder, query: &str| -> Vec<String> {
````

In `magcoupling-rs/src/gui/results_table.rs`, replace:

````rust
    }

    #[test]
    fn the_empty_table_names_the_filter_that_empties_it() {
        // The trace's text only when the trace itself marks none of the rows the failing filter
        // lets through, whatever the search; a search that hides the traced rows is the search's.
````

with:

````rust
    }

    #[test]
    fn a_row_carries_its_check_s_level_and_a_heading_the_worst_of_its_rows() {
        use crate::gui::test_support::short_magnets;
        let entries = table_entries();
        let row = |path: &str| &entries[entry_index(path).unwrap()];
        // The default design: the hot minimum fails (red); a result that is no check has none.
        let results = compute_all(&DesignInputs::default());
        assert_eq!(
            entry_level(&results, row("metal.hot_min_check")),
            Some(Level::Bad)
        );
        assert_eq!(entry_level(&results, row("model.pullout_Nm")), None);
        // Short magnets: the end effect greys the hot minimum, so its row has no level.
        let short = compute_all(&short_magnets());
        assert_eq!(entry_level(&short, row("metal.hot_min_check")), None);
        // Every row: its check's level, none for a row that is no check.
        for results in [&results, &short] {
            for entry in entries {
                let want = CHECKS
                    .contains(&entry.path.as_str())
                    .then(|| check_level(results, &entry.path))
                    .flatten();
                assert_eq!(entry_level(results, entry), want, "{}", entry.path);
            }
            // A heading's level: the worst of its rows' levels.
            for group in result_groups() {
                let levels = group
                    .rows
                    .iter()
                    .filter_map(|&i| entry_level(results, &entries[i]));
                assert_eq!(
                    worst_level(results, &group.rows),
                    levels.max(),
                    "{}",
                    group.label
                );
            }
        }
        let pair = [
            entry_index("model.pullout_Nm").unwrap(),
            entry_index("metal.hot_min_check").unwrap(),
        ];
        assert_eq!(worst_level(&results, &pair), Some(Level::Bad));
        assert_eq!(worst_level(&results, &pair[..1]), None);
        assert_eq!(worst_level(&results, &[]), None);
    }

    #[test]
    fn the_empty_table_names_the_filter_that_empties_it() {
        // The trace's text only when the trace itself marks none of the rows the failing filter
        // lets through, whatever the search; a search that hides the traced rows is the search's.
````


- [ ] **Step 2: Run the tests to see them fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-xlsx/magcoupling-rs/Cargo.toml --features app --lib -- spreadsheet a_group_lists_its_sections a_row_carries_its_check 2>&1 | grep -E "^error" | sort | uniq -c | sort -rn | head -n 12
```

Expected: FAIL to compile; the most frequent lines (the last line, not shown by the `head`, is `error: could not compile `magcoupling-rs` (lib test) due to 290 previous errors; 1 warning emitted`; further down, the helpers' tests fail too: `cannot find function `entry_level``, and `no method named` `plain_sections`, `advanced_sections`, `has_advanced` and `heading_in`):

```
     67 error[E0433]: failed to resolve: use of undeclared type `Cell`
     19 error[E0433]: failed to resolve: use of undeclared type `CellStyle`
     16 error[E0425]: cannot find function `sheets` in this scope
     14 error[E0433]: failed to resolve: use of undeclared type `DesignInputs`
     11 error[E0425]: cannot find function `value_cell` in this scope
      8 error[E0425]: cannot find value `RESULT_COLUMNS` in this scope
      8 error[E0425]: cannot find value `INPUT_COLUMNS` in this scope
      7 error[E0433]: failed to resolve: use of undeclared type `Value`
      7 error[E0425]: cannot find value `RESULTS_SHEET` in this scope
      6 error[E0425]: cannot find value `INPUTS_SHEET` in this scope
      5 error[E0433]: failed to resolve: use of undeclared type `InputOrder`
      5 error[E0425]: cannot find value `SUMMARY_SHEET` in this scope
```


- [ ] **Step 3: Write the layout and share the helpers it reuses**

In `magcoupling-rs/src/gui/spreadsheet.rs`, replace:

````rust
#[cfg(test)]
mod tests {
````

with:

````rust
//! The spreadsheet export's layout (user request 2026-10-03: "a download button for the
//! spreadsheet equivalent of the current layout of the website"): the design shown and its
//! results laid out as the page shows them, as values (no formulas, and nothing reads a
//! spreadsheet back into the site: decision X-1). [`sheets`] lays the workbook out as rows of
//! cells, one [`Sheet`] per sheet, in order (decision X-2): the Summary (the export, the share
//! link, the sizing state, the banners, the dashboard, the material warnings and the corrections
//! applied), the Inputs in the order the inputs side shows them (decision X-3), the Results by
//! physics chain as the results table groups them (decision X-5) and the Assumptions.
//! `gui::xlsx` writes the sheets as an .xlsx file.
//!
//! Every input and every result is on exactly one row: the Key design inputs are not repeated
//! above their groups, a column flags them instead (decision X-4). A number is a number cell; a
//! number that is not finite is the text the CSV export writes (`+inf`, `-inf`, `NaN`); a check's
//! level is its cell's style, which the writer fills with the badge's colour.

use crate::engine::assumptions;
use crate::engine::deviations::REGISTRY;
use crate::engine::explain::render;
use crate::engine::meta::{InputSet, ResultSet, Value};
use crate::engine::warnings::Severity;
use crate::gui::corrections::marks_values;
use crate::gui::dashboard::{
    Level, STORED_3D_LABEL, WARNINGS_HEADING, dashboard_lines, end_effect_banner, result_info,
    severity_level, warning_lines,
};
use crate::gui::input_ui::BLANK_TEXT;
use crate::gui::inputs::{
    ADVANCED_HEADING, InputCatalogue, InputEntry, InputGroup, InputOrder, InputSection, KEY_DESIGN,
};
use crate::gui::panel::{
    FREE_VARIABLE_LABEL, HEADING, SIZED_NOTE, TARGET_LABEL, assumptions_banner,
};
use crate::gui::readouts::registry;
use crate::gui::result_groups::{OTHER_RESULTS, result_groups};
use crate::gui::results_table::{
    Line, RUST_ONLY, ResultOrder, TableEntry, entry_level, exact_number, table_entries,
    table_lines, worst_level,
};
use crate::gui::sizing::{SizingMode, SizingState, TARGET_RANGE_INPUT, variable_label};
use crate::{DesignInputs, DesignResults};

/// The sheets' names, in order (decision X-2).
pub const SUMMARY_SHEET: &str = "Summary";
pub const INPUTS_SHEET: &str = "Inputs";
pub const RESULTS_SHEET: &str = "Results";
pub const ASSUMPTIONS_SHEET: &str = "Assumptions";

/// The Inputs sheet's column headers.
pub const INPUT_COLUMNS: [&str; 10] = [
    "Input",
    "Value",
    "Unit",
    "Workbook cell",
    "Path",
    "Changed from default",
    "Assumption",
    "Advanced",
    "Key design",
    "Note",
];

/// The Results sheet's column headers.
pub const RESULT_COLUMNS: [&str; 8] = [
    "Result",
    "Value",
    "Unit",
    "Check",
    "Workbook cell",
    "Path",
    "Corrected vs workbook",
    "Equation",
];

/// The column headers of the Summary's dashboard rows.
pub const DASHBOARD_COLUMNS: [&str; 8] = [
    "Dashboard",
    "Value",
    "Unit",
    "Check",
    "Workbook cell",
    "Path",
    "Corrected vs workbook",
    "Note",
];

/// The Assumptions sheet's column headers.
pub const ASSUMPTION_COLUMNS: [&str; 8] = [
    "Assumption",
    "Path",
    "Value",
    "Unit",
    "Workbook default",
    "Changed from default",
    "Rationale",
    "Source",
];

/// The Summary's row labels.
pub const EXPORTED: &str = "Exported (UTC)";
pub const APP: &str = "App";
pub const SHARE_LINK: &str = "Share link (paste into a browser)";
pub const SIZING_MODE: &str = "Sizing mode";
pub const SIZING_OUTCOME: &str = "Sizing outcome";

/// The Summary's heading over the corrections applied.
pub const CORRECTIONS_HEADING: &str = "Corrections applied (every approved correction is on)";

/// The Summary's line when no material warning fires.
pub const NO_WARNING: &str = "No material warning fires.";

/// The note of a dashboard row the page greys (audit M9).
pub const GREYED_NOTE: &str = "Greyed on the page: computed from the pull-out (see the banner)";

/// A flag column's text when the flag is set (blank when not).
pub const YES: &str = "yes";

/// The columns' widths [characters].
const SUMMARY_WIDTHS: [f64; 8] = [44.0, 24.0, 10.0, 8.0, 24.0, 44.0, 12.0, 44.0];
const INPUT_WIDTHS: [f64; 10] = [44.0, 16.0, 10.0, 24.0, 44.0, 11.0, 11.0, 10.0, 10.0, 30.0];
const RESULT_WIDTHS: [f64; 8] = [44.0, 16.0, 10.0, 8.0, 24.0, 44.0, 12.0, 70.0];
const ASSUMPTION_WIDTHS: [f64; 8] = [32.0, 40.0, 14.0, 10.0, 14.0, 11.0, 70.0, 50.0];

/// What a cell holds.
#[derive(Clone, Debug, PartialEq)]
pub enum CellValue {
    Empty,
    /// A finite number (a number that is not finite is text: [`value_cell`]).
    Number(f64),
    /// Never empty: an empty text is [`CellValue::Empty`] ([`Cell::text`]).
    Text(String),
    /// A UTC date and time [s since the Unix epoch].
    Time(i64),
}

/// How a cell looks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CellStyle {
    #[default]
    Plain,
    /// The Summary's title.
    Title,
    /// A column header (the frozen row).
    Header,
    /// A group heading: an input group, a result chain, "Other results", a Summary block.
    Group,
    /// A heading inside a group: a section, "Advanced", a package group of the other results.
    Section,
    /// A check's level: the cell is filled with the badge's colour.
    Level(Level),
}

/// One cell.
#[derive(Clone, Debug, PartialEq)]
pub struct Cell {
    pub value: CellValue,
    pub style: CellStyle,
}

impl Cell {
    /// An empty cell.
    pub fn empty() -> Self {
        Self {
            value: CellValue::Empty,
            style: CellStyle::Plain,
        }
    }

    /// A number cell.
    pub fn number(x: f64) -> Self {
        Self {
            value: CellValue::Number(x),
            style: CellStyle::Plain,
        }
    }

    /// A text cell; an empty cell for an empty text.
    pub fn text(text: impl Into<String>) -> Self {
        let text = text.into();
        Self {
            value: if text.is_empty() {
                CellValue::Empty
            } else {
                CellValue::Text(text)
            },
            style: CellStyle::Plain,
        }
    }

    /// The cell with `style`.
    pub fn styled(mut self, style: CellStyle) -> Self {
        self.style = style;
        self
    }

    /// The cell's text; empty for a cell that holds no text.
    pub fn as_text(&self) -> &str {
        match &self.value {
            CellValue::Text(text) => text,
            _ => "",
        }
    }
}

/// One sheet: its name, its columns' widths, the rows frozen at its top, whether its first row
/// carries Excel's filter buttons, and its rows (a row may be shorter than the sheet is wide).
#[derive(Clone, Debug, PartialEq)]
pub struct Sheet {
    pub name: &'static str,
    /// Each column's width, left to right [characters].
    pub widths: &'static [f64],
    pub frozen_rows: u32,
    /// The first row (the column headers) filters every column below it.
    pub autofilter: bool,
    pub rows: Vec<Vec<Cell>>,
}

/// What the spreadsheet shows: the design on the page when the button was pressed.
#[derive(Clone, Debug)]
pub struct Snapshot<'a> {
    /// The design shown: in Torque -> Magnets the inputs with the free variable at the value
    /// shown.
    pub inputs: &'a DesignInputs,
    /// Its results.
    pub results: &'a DesignResults,
    pub sizing: SizingState,
    /// The sizing outcome line, in Torque -> Magnets.
    pub sizing_status: Option<String>,
    /// The order of the inputs side (decision O-1).
    pub input_order: InputOrder,
    /// The link that reopens the design on the site.
    pub share_link: String,
    /// When it was exported [s since the Unix epoch].
    pub exported_unix_s: i64,
}

/// The cell of an engine value: a number cell for a finite number or an integer, the CSV
/// export's text for a number that is not finite (`+inf`, `-inf`, `NaN`), text as it is, and an
/// empty cell for none.
pub fn value_cell(value: &Value) -> Cell {
    match value {
        Value::Num(x) if x.is_finite() => Cell::number(*x),
        Value::Num(x) => Cell::text(exact_number(*x)),
        Value::Int(i) => Cell::number(*i as f64),
        Value::Text(text) => Cell::text(text.clone()),
        Value::None => Cell::empty(),
    }
}

/// The cell of a check's level: the badge's word, filled with its colour; empty for none.
pub fn level_cell(level: Option<Level>) -> Cell {
    level.map_or_else(Cell::empty, |level| {
        Cell::text(level.word()).styled(CellStyle::Level(level))
    })
}

/// The workbook, sheet by sheet, in order (decision X-2).
pub fn sheets(snapshot: &Snapshot) -> Vec<Sheet> {
    vec![
        summary_sheet(snapshot),
        inputs_sheet(snapshot),
        results_sheet(snapshot.results),
        assumptions_sheet(snapshot.inputs),
    ]
}

/// A row of one heading cell.
fn heading_row(text: &str, style: CellStyle) -> Vec<Cell> {
    vec![Cell::text(text).styled(style)]
}

/// A row of column headers.
fn header_row(columns: &[&str]) -> Vec<Cell> {
    columns
        .iter()
        .map(|column| Cell::text(*column).styled(CellStyle::Header))
        .collect()
}

/// A flag column's cell: [`YES`] when set, empty when not.
fn flag(on: bool) -> Cell {
    if on { Cell::text(YES) } else { Cell::empty() }
}

/// A workbook cell column's cell: the cell, or [`RUST_ONLY`] for none.
fn workbook_cell(cell: Option<&str>) -> Cell {
    Cell::text(cell.unwrap_or(RUST_ONLY))
}

/// The Summary: the title, when and by what it was exported, the share link, the sizing state,
/// the end-effect and assumptions banners when they show, then the dashboard's rows, the material
/// warnings and the corrections applied. Nothing is frozen: its blocks are read top to bottom.
fn summary_sheet(snapshot: &Snapshot) -> Sheet {
    let results = snapshot.results;
    let sizing = snapshot.sizing;
    let mut rows = vec![
        heading_row(HEADING, CellStyle::Title),
        vec![
            Cell::text(EXPORTED),
            Cell {
                value: CellValue::Time(snapshot.exported_unix_s),
                style: CellStyle::Plain,
            },
        ],
        vec![
            Cell::text(APP),
            Cell::text(format!("magcoupling-rs {}", env!("CARGO_PKG_VERSION"))),
        ],
        vec![
            Cell::text(SHARE_LINK),
            Cell::text(snapshot.share_link.clone()),
        ],
        vec![Cell::text(SIZING_MODE), Cell::text(sizing.mode.label())],
    ];
    if sizing.mode == SizingMode::TorqueToMagnets {
        let target_unit = InputCatalogue::get()
            .entry(TARGET_RANGE_INPUT)
            .map_or("", |entry| entry.meta.unit);
        rows.push(vec![
            Cell::text(FREE_VARIABLE_LABEL),
            Cell::text(variable_label(sizing.variable)),
        ]);
        rows.push(vec![
            Cell::text(TARGET_LABEL),
            value_cell(&Value::Num(sizing.target_Nm)),
            Cell::text(target_unit),
        ]);
        if let Some(status) = &snapshot.sizing_status {
            rows.push(vec![Cell::text(SIZING_OUTCOME), Cell::text(status.clone())]);
        }
    }
    if let Some(banner) = end_effect_banner(results) {
        rows.push(heading_row(&banner, CellStyle::Level(Level::Bad)));
    }
    if let Some(banner) = assumptions_banner(snapshot.inputs) {
        rows.push(heading_row(&banner, CellStyle::Level(Level::Caution)));
    }
    rows.push(Vec::new());
    rows.push(header_row(&DASHBOARD_COLUMNS));
    for line in dashboard_lines(results) {
        let info = result_info(line.path).expect("a dashboard row is a result");
        let note = if line.greyed {
            GREYED_NOTE
        } else if line.stored_3d {
            STORED_3D_LABEL
        } else {
            ""
        };
        rows.push(vec![
            Cell::text(line.label),
            value_cell(&results.get(line.path).unwrap_or(Value::None)),
            Cell::text(info.meta.unit),
            level_cell(line.level),
            workbook_cell(info.cell.as_deref()),
            Cell::text(line.path),
            Cell::text(line.marker.clone()),
            Cell::text(note),
        ]);
    }
    rows.push(Vec::new());
    rows.push(heading_row(WARNINGS_HEADING, CellStyle::Group));
    let warnings = warning_lines(results);
    if warnings.is_empty() {
        rows.push(vec![Cell::text(NO_WARNING)]);
    }
    for (rule, text) in warnings {
        let severity = match rule.severity {
            Severity::Warning => "Warning",
            Severity::Caution => "Caution",
        };
        rows.push(vec![
            Cell::text(severity).styled(CellStyle::Level(severity_level(rule.severity))),
            Cell::text(text),
        ]);
    }
    rows.push(Vec::new());
    rows.push(heading_row(CORRECTIONS_HEADING, CellStyle::Group));
    for deviation in REGISTRY.iter().filter(|deviation| marks_values(deviation)) {
        rows.push(vec![
            Cell::text(deviation.id.to_string()),
            Cell::text(deviation.title),
        ]);
    }
    Sheet {
        name: SUMMARY_SHEET,
        widths: &SUMMARY_WIDTHS,
        frozen_rows: 0,
        autofilter: false,
        rows,
    }
}

/// The Inputs, in the order the inputs side shows them (decision X-3), drawn by the page's own
/// rules: each group's heading, its plain sections ([`InputGroup::plain_sections`]), then in the
/// workflow order its Advanced heading and the advanced sections
/// ([`InputGroup::advanced_sections`]).
fn inputs_sheet(snapshot: &Snapshot) -> Sheet {
    let sized = (snapshot.sizing.mode == SizingMode::TorqueToMagnets)
        .then(|| snapshot.sizing.variable.path());
    let mut rows = vec![header_row(&INPUT_COLUMNS)];
    for group in InputCatalogue::get().groups_in(snapshot.input_order) {
        rows.push(heading_row(group.label, CellStyle::Group));
        for section in group.plain_sections() {
            section_rows(&mut rows, group, section, snapshot.inputs, sized);
        }
        if group.has_advanced() {
            rows.push(heading_row(ADVANCED_HEADING, CellStyle::Section));
            for section in group.advanced_sections() {
                section_rows(&mut rows, group, section, snapshot.inputs, sized);
            }
        }
    }
    Sheet {
        name: INPUTS_SHEET,
        widths: &INPUT_WIDTHS,
        frozen_rows: 1,
        autofilter: true,
        rows,
    }
}

/// One section's rows: its heading ([`InputSection::heading_in`]: none for the group's own
/// section), then its inputs.
fn section_rows(
    rows: &mut Vec<Vec<Cell>>,
    group: &InputGroup,
    section: &InputSection,
    inputs: &DesignInputs,
    sized: Option<&str>,
) {
    if let Some(heading) = section.heading_in(group) {
        rows.push(heading_row(heading, CellStyle::Section));
    }
    for entry in &section.entries {
        rows.push(input_row(entry, section.advanced, inputs, sized));
    }
}

/// One input's row: label, value, unit, workbook cell, path, its flags, and a note naming a
/// selector's choice, a blank optional input or empty text (a part name left out for manual
/// magnets), or the free variable Torque -> Magnets set (`sized`).
fn input_row(
    entry: &InputEntry,
    advanced: bool,
    inputs: &DesignInputs,
    sized: Option<&str>,
) -> Vec<Cell> {
    let meta = entry.meta;
    let value = inputs.get(&entry.path).unwrap_or(Value::None);
    let mut notes = Vec::new();
    if sized == Some(entry.path.as_str()) {
        notes.push(SIZED_NOTE.to_owned());
    }
    match &value {
        Value::Int(code) if !meta.choices.is_empty() => {
            if let Some((_, text)) = meta.choices.iter().find(|(c, _)| c == code) {
                notes.push(format!("{code} = {text}"));
            }
        }
        Value::None => notes.push(BLANK_TEXT.to_owned()),
        Value::Text(text) if text.is_empty() => notes.push(BLANK_TEXT.to_owned()),
        _ => {}
    }
    vec![
        Cell::text(meta.label),
        value_cell(&value),
        Cell::text(meta.unit),
        workbook_cell(meta.cell),
        Cell::text(entry.path.clone()),
        flag(value != entry.default),
        flag(meta.assumption),
        flag(advanced),
        flag(KEY_DESIGN.contains(&entry.path.as_str())),
        Cell::text(notes.join("; ")),
    ]
}

/// The Results by physics chain, as the results table groups them (decision X-5): the table's
/// own grouped lines ([`table_lines`], every row, every group open), so the headings are the
/// table's, in its order (the package groups under "Other results"; a group with no row is left
/// out, as the table leaves it out). Each group's heading carries the worst level of its checks,
/// each result its row ([`result_row`]).
fn results_sheet(results: &DesignResults) -> Sheet {
    let entries = table_entries();
    let all: Vec<usize> = (0..entries.len()).collect();
    let mut rows = vec![header_row(&RESULT_COLUMNS)];
    for line in table_lines(&all, ResultOrder::Grouped, None, &|_| true) {
        match line {
            Line::OtherResults => rows.push(heading_row(OTHER_RESULTS, CellStyle::Group)),
            Line::Group { group: index, .. } => {
                let group = &result_groups()[index];
                let style = if group.other {
                    CellStyle::Section
                } else {
                    CellStyle::Group
                };
                let mut heading = heading_row(group.label, style);
                heading.extend([
                    Cell::empty(),
                    Cell::empty(),
                    level_cell(worst_level(results, &group.rows)),
                ]);
                rows.push(heading);
            }
            Line::Row(index) => rows.push(result_row(&entries[index], results)),
        }
    }
    Sheet {
        name: RESULTS_SHEET,
        widths: &RESULT_WIDTHS,
        frozen_rows: 1,
        autofilter: true,
        rows,
    }
}

/// One result's row: label, value, unit, the level of its row's badge ([`entry_level`]),
/// workbook cell, path, corrections, and the equation as plain text where a record exists.
fn result_row(entry: &TableEntry, results: &DesignResults) -> Vec<Cell> {
    let equation = registry()
        .equation_for(&entry.path)
        .map_or_else(String::new, |eq| {
            render::plain(registry(), &eq.symbol, &eq.formula)
        });
    vec![
        Cell::text(entry.info.meta.label),
        value_cell(&results.get(&entry.path).unwrap_or(Value::None)),
        Cell::text(entry.info.meta.unit),
        level_cell(entry_level(results, entry)),
        workbook_cell(entry.info.cell.as_deref()),
        Cell::text(entry.path.clone()),
        Cell::text(entry.marker.clone()),
        Cell::text(equation),
    ]
}

/// The Assumptions, as the Assumptions view lists them: one row per input of each assumption,
/// with its workbook default, whether it differs, the rationale and the source.
fn assumptions_sheet(inputs: &DesignInputs) -> Sheet {
    let mut rows = vec![header_row(&ASSUMPTION_COLUMNS)];
    for state in assumptions::states(inputs) {
        let assumption = state.assumption;
        let values = state.values.iter().zip(&state.defaults);
        for (path, (value, default)) in assumption.paths.iter().zip(values) {
            rows.push(vec![
                Cell::text(assumption.label),
                Cell::text(*path),
                value_cell(value),
                Cell::text(state.unit),
                value_cell(default),
                flag(value != default),
                Cell::text(assumption.rationale),
                Cell::text(assumption.source),
            ]);
        }
    }
    Sheet {
        name: ASSUMPTIONS_SHEET,
        widths: &ASSUMPTION_WIDTHS,
        frozen_rows: 1,
        autofilter: true,
        rows,
    }
}

#[cfg(test)]
mod tests {
````

In `magcoupling-rs/src/gui/corrections.rs`, replace:

````rust
    }
}

/// Whether a correction marks values: applied and changing what the engine computes.
fn marks_values(deviation: &Deviation) -> bool {
    deviation.status == DeviationStatus::Applied && deviation.class == DeviationClass::Engine
}
````

with:

````rust
    }
}

/// Whether a correction marks values: applied and changing what the engine computes (the
/// spreadsheet's Summary lists these as the corrections applied).
pub(crate) fn marks_values(deviation: &Deviation) -> bool {
    deviation.status == DeviationStatus::Applied && deviation.class == DeviationClass::Engine
}
````

In `magcoupling-rs/src/gui/inputs.rs`, replace:

````rust
    pub sections: Vec<InputSection>,
}

/// Every input, arranged for the left side of the panel.
#[derive(Clone, Debug, PartialEq)]
pub struct InputCatalogue {
````

with:

````rust
    pub sections: Vec<InputSection>,
}

impl InputSection {
    /// The heading this section is drawn under in `group`: its label, or `None` for the group's
    /// own section, drawn without one. The inputs side, the filter's runs and the spreadsheet
    /// all head a section so.
    pub fn heading_in(&self, group: &InputGroup) -> Option<&'static str> {
        (self.id != group.name).then_some(self.label)
    }
}

impl InputGroup {
    /// The sections drawn under the group's heading, in order: the inputs side and the
    /// spreadsheet draw these first.
    pub fn plain_sections(&self) -> impl Iterator<Item = &InputSection> {
        self.sections.iter().filter(|section| !section.advanced)
    }

    /// The sections drawn under the group's [`ADVANCED_HEADING`] (workflow order only), in
    /// order, after the plain ones.
    pub fn advanced_sections(&self) -> impl Iterator<Item = &InputSection> {
        self.sections.iter().filter(|section| section.advanced)
    }

    /// Whether the group has an [`ADVANCED_HEADING`]: a section under it.
    pub fn has_advanced(&self) -> bool {
        self.advanced_sections().next().is_some()
    }
}

/// Every input, arranged for the left side of the panel.
#[derive(Clone, Debug, PartialEq)]
pub struct InputCatalogue {
````

In `magcoupling-rs/src/gui/inputs.rs`, replace:

````rust
}

impl SectionMatches<'_> {
    /// The run's heading: the group's label, then the section's unless it is the group's own.
    pub fn heading(&self) -> String {
        if self.section.id == self.group.name {
            self.group.label.to_owned()
        } else {
            format!("{} / {}", self.group.label, self.section.label)
        }
    }
}
````

with:

````rust
}

impl SectionMatches<'_> {
    /// The run's heading: the group's label, then the section's unless it is the group's own
    /// ([`InputSection::heading_in`]).
    pub fn heading(&self) -> String {
        match self.section.heading_in(self.group) {
            Some(section) => format!("{} / {section}", self.group.label),
            None => self.group.label.to_owned(),
        }
    }
}
````

In `magcoupling-rs/src/gui/results_table.rs`, replace:

````rust
/// The search box's hint.
pub const SEARCH_HINT: &str = "Search label, path or cell";

/// The failing filter's checkbox (decision O-7).
pub const FAILING_ONLY: &str = "Failing checks only";
````

with:

````rust
/// The search box's hint.
pub const SEARCH_HINT: &str = "Search label, path or cell";

/// The workbook cell column of a Rust-only result (no workbook cell).
pub const RUST_ONLY: &str = "Rust-only";

/// The failing filter's checkbox (decision O-7).
pub const FAILING_ONLY: &str = "Failing checks only";
````

In `magcoupling-rs/src/gui/results_table.rs`, replace:

````rust
            lines
        }
    }
}

/// What the table says when its filters leave no line: [`NOTHING_TRACED`] when the trace filter
````

with:

````rust
            lines
        }
    }
}

/// The level of the badge `entry`'s row carries for `results`, in the table and the
/// spreadsheet: its check's level ([`check_level`]); `None` for a row that is no check and for
/// a check without a level (one the end effect greys).
pub fn entry_level(results: &DesignResults, entry: &TableEntry) -> Option<Level> {
    entry
        .check
        .then(|| check_level(results, &entry.path))
        .flatten()
}

/// The worst level of the checks among `rows` (indices into [`table_entries`]) for `results`: a
/// group heading's badge in the table and the spreadsheet; `None` when none of them is a check
/// with a level.
pub fn worst_level(results: &DesignResults, rows: &[usize]) -> Option<Level> {
    let entries = table_entries();
    rows.iter()
        .filter_map(|&index| entry_level(results, &entries[index]))
        .max()
}

/// What the table says when its filters leave no line: [`NOTHING_TRACED`] when the trace filter
````

In `magcoupling-rs/src/gui/results_table.rs`, replace:

````rust
                            };
                            // The worst level of the group's checks: a closed group shows that
                            // one of them fails.
                            let level = heading
                                .rows
                                .iter()
                                .map(|&index| &entries[index])
                                .filter(|entry| entry.check)
                                .filter_map(|entry| check_level(results, &entry.path))
                                .max();
                            let line = HeadingLine {
                                text: &text,
                                open,
````

with:

````rust
                            };
                            // The worst level of the group's checks: a closed group shows that
                            // one of them fails.
                            let level = worst_level(results, &heading.rows);
                            let line = HeadingLine {
                                text: &text,
                                open,
````

In `magcoupling-rs/src/gui/results_table.rs`, replace:

````rust
                        }
                        Line::Row(index) => {
                            let entry = &entries[index];
                            let level = entry
                                .check
                                .then(|| check_level(results, &entry.path))
                                .flatten();
                            row_ui(ui, entry, results, level, row_height, widths, readouts);
                        }
                    }
````

with:

````rust
                        }
                        Line::Row(index) => {
                            let entry = &entries[index];
                            let level = entry_level(results, entry);
                            row_ui(ui, entry, results, level, row_height, widths, readouts);
                        }
                    }
````

In `magcoupling-rs/src/gui/results_table.rs`, replace:

````rust
        cell(
            ui,
            workbook,
            entry.info.cell.as_deref().unwrap_or("Rust-only"),
            None,
        );
        cell(ui, marker, &entry.marker, None);
````

with:

````rust
        cell(
            ui,
            workbook,
            entry.info.cell.as_deref().unwrap_or(RUST_ONLY),
            None,
        );
        cell(ui, marker, &entry.marker, None);
````

In `magcoupling-rs/src/gui/panel.rs`, replace:

````rust
            }
        });
        // Spec Addendum A3: the banner while any assumption differs from its workbook default.
        let modified = assumptions::modified(&self.inputs);
        let banner = (!modified.is_empty()).then(|| {
            let names: Vec<&str> = modified.iter().map(|a| a.label).collect();
            format!("{ASSUMPTIONS_MODIFIED}: {}", names.join(", "))
        });
        if banner != self.banner {
            self.banner.clone_from(&banner);
            ui.ctx().request_repaint();
````

with:

````rust
            }
        });
        // Spec Addendum A3: the banner while any assumption differs from its workbook default.
        let banner = assumptions_banner(&self.inputs);
        if banner != self.banner {
            self.banner.clone_from(&banner);
            ui.ctx().request_repaint();
````

In `magcoupling-rs/src/gui/panel.rs`, replace:

````rust
                        .default_open(false)
                        .open(open_group.then_some(true))
                        .show(ui, |ui| {
                            for section in group.sections.iter().filter(|s| !s.advanced) {
                                self.section_ui(ui, group, section, readouts);
                            }
                            if group.sections.iter().any(|s| s.advanced) {
                                egui::CollapsingHeader::new(ADVANCED_HEADING)
                                    .id_salt(("advanced", &group.name))
                                    .default_open(false)
                                    .open(open_advanced.then_some(true))
                                    .show(ui, |ui| {
                                        for section in group.sections.iter().filter(|s| s.advanced)
                                        {
                                            self.section_ui(ui, group, section, readouts);
                                        }
                                    });
````

with:

````rust
                        .default_open(false)
                        .open(open_group.then_some(true))
                        .show(ui, |ui| {
                            for section in group.plain_sections() {
                                self.section_ui(ui, group, section, readouts);
                            }
                            if group.has_advanced() {
                                egui::CollapsingHeader::new(ADVANCED_HEADING)
                                    .id_salt(("advanced", &group.name))
                                    .default_open(false)
                                    .open(open_advanced.then_some(true))
                                    .show(ui, |ui| {
                                        for section in group.advanced_sections() {
                                            self.section_ui(ui, group, section, readouts);
                                        }
                                    });
````

In `magcoupling-rs/src/gui/panel.rs`, replace:

````rust
            });
    }

    /// One section of a group: its heading (none for the group's own section), then its rows.
    fn section_ui(
        &mut self,
        ui: &mut egui::Ui,
````

with:

````rust
            });
    }

    /// One section of a group: its heading ([`InputSection::heading_in`]: none for the group's
    /// own section), then its rows.
    fn section_ui(
        &mut self,
        ui: &mut egui::Ui,
````

In `magcoupling-rs/src/gui/panel.rs`, replace:

````rust
        section: &'static InputSection,
        readouts: &Readouts,
    ) {
        if section.id != group.name {
            ui.add_space(4.0);
            ui.strong(section.label);
        }
        for entry in &section.entries {
            let widget = self.input_row_ui(ui, entry, readouts);
````

with:

````rust
        section: &'static InputSection,
        readouts: &Readouts,
    ) {
        if let Some(heading) = section.heading_in(group) {
            ui.add_space(4.0);
            ui.strong(heading);
        }
        for entry in &section.entries {
            let widget = self.input_row_ui(ui, entry, readouts);
````

In `magcoupling-rs/src/gui/panel.rs`, replace:

````rust
            _ => None,
        }
    }
}

/// Draws `add` in a child of `ui` (salted `id_salt`) confined to the space left in `ui`: the
````

with:

````rust
            _ => None,
        }
    }
}

/// The assumptions banner (spec Addendum A3) of `inputs`: [`ASSUMPTIONS_MODIFIED`] and the names
/// of the assumptions that differ from their workbook default; `None` while none does. The header
/// shows it, and the spreadsheet's Summary.
pub(crate) fn assumptions_banner(inputs: &DesignInputs) -> Option<String> {
    let modified = assumptions::modified(inputs);
    (!modified.is_empty()).then(|| {
        let names: Vec<&str> = modified.iter().map(|a| a.label).collect();
        format!("{ASSUMPTIONS_MODIFIED}: {}", names.join(", "))
    })
}

/// Draws `add` in a child of `ui` (salted `id_salt`) confined to the space left in `ui`: the
````


- [ ] **Step 4: Run the tests to see them pass**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-xlsx/magcoupling-rs/Cargo.toml --features app --lib -- spreadsheet a_group_lists_its_sections a_row_carries_its_check 2>&1 | grep -E "^test |test result"
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-xlsx/magcoupling-rs/Cargo.toml --features app --lib 2>&1 | grep -E "test result"
```

Expected: the 12 `gui::spreadsheet::tests`, `gui::inputs::tests::a_group_lists_its_sections_and_their_headings_as_the_page_draws_them` and `gui::results_table::tests::a_row_carries_its_check_s_level_and_a_heading_the_worst_of_its_rows` `ok` and `test result: ok. 14 passed; 0 failed; 0 ignored; 0 measured; 498 filtered out`; the library `test result: ok. 512 passed`.

- [ ] **Step 5: Format, lint and gate**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-xlsx/magcoupling-rs/Cargo.toml --check
cargo clippy --manifest-path C:/Users/Cole/source/repos/lsim-mag-xlsx/magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings 2>&1 | tail -n 1
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-xlsx/linkage-sim-rs/scripts/gate.sh 2>&1 | tail -n 3
git -C C:/Users/Cole/source/repos/lsim-mag-xlsx checkout -- docs/chebyshev_lambda
```

Expected: `fmt --check` prints nothing; clippy ends `Finished`; the gate ends `GATE PASS` with no `SKIP gate` line.

- [ ] **Step 6: Commit**

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-xlsx add magcoupling-rs/src/gui/spreadsheet.rs magcoupling-rs/src/gui/mod.rs magcoupling-rs/src/gui/corrections.rs magcoupling-rs/src/gui/inputs.rs magcoupling-rs/src/gui/results_table.rs magcoupling-rs/src/gui/panel.rs
git -C C:/Users/Cole/source/repos/lsim-mag-xlsx commit -m "feat(magcoupling-rs): the spreadsheet's layout, the page as sheets of values (decisions X-2 to X-6, X-11, X-12)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9"
git -C C:/Users/Cole/source/repos/lsim-mag-xlsx status --short
```

Expected: one commit; no status lines.

### Task 3: The .xlsx writer (`gui::xlsx`, rust_xlsxwriter)

**Model:** `sonnet` (exact code; CLAUDE.md section 5). Escalate a retry to the session model.

**Files:**
- Modify: `magcoupling-rs/Cargo.toml` (rust_xlsxwriter `=0.99.1` and js-sys in feature `gui`; its `wasm` feature in the wasm32 table; calamine and zip dev-dependencies)
- Modify: `magcoupling-rs/src/gui/test_support.rs` (`read_xlsx`, `xlsx_part`)
- Create: `magcoupling-rs/src/gui/xlsx.rs` (tests first, then the module above them)
- Modify: `magcoupling-rs/src/gui/mod.rs` (`pub mod xlsx;`)
- Modify: `magcoupling-rs/src/gui/session.rs`, `linkage-sim-rs/src/gui/state/file_io.rs` (a test each: a share link written at `49fbd3f` still decodes; they guard the flate2 backend switch that Step 1's dependency brings, so they are never red: the session test runs in Step 4, the linkage one in Step 5)
- Modify: `linkage-sim-rs/scripts/gate.sh` (gate 11 checks `rust_xlsxwriter`)
- Modify: `magcoupling-rs/README.md` (the feature table's `gui` row)
- Modify (by cargo): `magcoupling-rs/Cargo.lock`, `linkage-sim-rs/Cargo.lock`

**Interfaces:**
- Consumes: `sheets`, `Sheet`, `Cell`, `CellValue`, `CellStyle`, `Snapshot` (Task 2); `Level::color(&egui::Visuals)` (dashboard); `HEADING` (panel).
- Produces: `pub fn spreadsheet_bytes(snapshot: &Snapshot) -> Result<Vec<u8>, String>`; `pub fn xlsx_bytes(sheets: &[Sheet], created_unix_s: i64) -> Result<Vec<u8>, rust_xlsxwriter::XlsxError>`; `pub fn cell_text(text: &str) -> String`; `pub fn fill(level: Level) -> rust_xlsxwriter::Color`; `pub const GROUP_FILL: rust_xlsxwriter::Color` (E7E6E6); `pub fn now_unix_s() -> i64`; `pub const XLSX_FILE_NAME: &str = "magcoupling-results.xlsx"`; `pub const XLSX_MIME: &str = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"`; `pub const MAX_CELL_CHARS: usize = 32_767`; `pub const TIME_FORMAT: &str`; test helpers `test_support::read_xlsx(&[u8]) -> Vec<(String, Vec<Vec<calamine::Data>>)>` and `test_support::xlsx_part(&[u8], &str) -> String`.

- [ ] **Step 1: Add the dependencies and write the failing tests**

In `magcoupling-rs/Cargo.toml`, replace:

````toml
# windowing, no eframe, no file dialogs. serde_json, base64 and flate2 write
# and read design files, share links and the results export; log reports the
# sizing outcome (the host chooses the logger: the browser console on the web);
# egui_plot draws the plot tabs.
gui = [
    "dep:egui",
    "dep:egui_plot",
````

with:

````toml
# windowing, no eframe, no file dialogs. serde_json, base64 and flate2 write
# and read design files, share links and the results export; log reports the
# sizing outcome (the host chooses the logger: the browser console on the web);
# egui_plot draws the plot tabs; rust_xlsxwriter writes the spreadsheet export,
# and js-sys gives it the time on the web (wasm32 only).
gui = [
    "dep:egui",
    "dep:egui_plot",
````

In `magcoupling-rs/Cargo.toml`, replace:

````toml
    "dep:serde_json",
    "dep:base64",
    "dep:flate2",
]
# The standalone app: `app::MagcouplingApp` and the binaries `magcoupling-app`
# (native) and `magcoupling-web` (wasm32, eframe WebRunner + wasm-bindgen).
````

with:

````toml
    "dep:serde_json",
    "dep:base64",
    "dep:flate2",
    "dep:rust_xlsxwriter",
    "dep:js-sys",
]
# The standalone app: `app::MagcouplingApp` and the binaries `magcoupling-app`
# (native) and `magcoupling-web` (wasm32, eframe WebRunner + wasm-bindgen).
````

In `magcoupling-rs/Cargo.toml`, replace:

````toml
# File dialogs of the standalone app (feature app), the version linkage-sim-rs
# uses: native save and open, and the web file picker.
rfd = { version = "0.15", optional = true }

[target.'cfg(not(target_arch = "wasm32"))'.dependencies]
env_logger = { version = "0.11", optional = true }
````

with:

````toml
# File dialogs of the standalone app (feature app), the version linkage-sim-rs
# uses: native save and open, and the web file picker.
rfd = { version = "0.15", optional = true }
# The spreadsheet export (feature gui, decision X-9): pinned exactly, and gate 11
# checks that both lock files agree (linkage-sim-rs carries the panel). Its zip
# brings flate2's zlib-rs backend and zopfli. On wasm32 the target table below
# adds its `wasm` feature: without it the workbook's creation time calls
# SystemTime::now(), which panics in the browser.
rust_xlsxwriter = { version = "=0.99.1", optional = true, default-features = false }

[target.'cfg(not(target_arch = "wasm32"))'.dependencies]
env_logger = { version = "0.11", optional = true }
````

In `magcoupling-rs/Cargo.toml`, replace:

````toml
    "Window",
] }
js-sys = { version = "0.3", optional = true }

[dev-dependencies]
magcoupling-rs = { path = ".", features = ["workbook-parity"] }
# float_roundtrip: parse JSON floats exactly (the default parser can be 1 ULP off).
serde_json = { version = "1", features = ["float_roundtrip"] }

# The standalone app (feature app). magcoupling-app is native only;
# magcoupling-web is the wasm32 entry (on a desktop it opens the same window).
````

with:

````toml
    "Window",
] }
js-sys = { version = "0.3", optional = true }
# The spreadsheet export's clock in the browser (js_sys::Date): see [dependencies].
rust_xlsxwriter = { version = "=0.99.1", optional = true, default-features = false, features = [
    "wasm",
] }

[dev-dependencies]
magcoupling-rs = { path = ".", features = ["workbook-parity"] }
# float_roundtrip: parse JSON floats exactly (the default parser can be 1 ULP off).
serde_json = { version = "1", features = ["float_roundtrip"] }
# Read the spreadsheet export back in tests: calamine its cells, zip its XML parts
# (styles, frozen panes, widths). The zip rust_xlsxwriter and calamine use.
calamine = "0.36"
zip = { version = "8", default-features = false, features = ["deflate"] }

# The standalone app (feature app). magcoupling-app is native only;
# magcoupling-web is the wasm32 entry (on a desktop it opens the same window).
````

In `magcoupling-rs/src/gui/test_support.rs`, replace:

````rust
            _ => None,
        })
}
````

with:

````rust
            _ => None,
        })
}

/// Every sheet of an .xlsx file as calamine reads it: its name and its rows from A1 (calamine's
/// range starts at the first cell used; every sheet of the export uses A1), each row as wide as
/// the sheet's widest, `Data::Empty` where nothing was written.
pub(crate) fn read_xlsx(bytes: &[u8]) -> Vec<(String, Vec<Vec<calamine::Data>>)> {
    use calamine::Reader;
    let mut workbook: calamine::Xlsx<_> =
        calamine::open_workbook_from_rs(std::io::Cursor::new(bytes.to_vec()))
            .expect("an .xlsx file calamine reads");
    workbook
        .sheet_names()
        .into_iter()
        .map(|name| {
            let range = workbook.worksheet_range(&name).expect("the sheet reads");
            assert_eq!(
                range.start().unwrap_or((0, 0)),
                (0, 0),
                "{name} starts at A1"
            );
            let rows = range.rows().map(<[calamine::Data]>::to_vec).collect();
            (name, rows)
        })
        .collect()
}

/// The text of the part `name` of an .xlsx file (a zip archive), e.g. `xl/styles.xml`.
pub(crate) fn xlsx_part(bytes: &[u8], name: &str) -> String {
    use std::io::Read;
    let mut archive =
        zip::ZipArchive::new(std::io::Cursor::new(bytes)).expect("an .xlsx file is a zip");
    let mut part = archive
        .by_name(name)
        .unwrap_or_else(|_| panic!("no part {name}"));
    let mut text = String::new();
    part.read_to_string(&mut text)
        .expect("an XML part is UTF-8");
    text
}
````

Create `magcoupling-rs/src/gui/xlsx.rs`:

````rust
#[cfg(test)]
mod tests {
    use calamine::Data;

    use super::*;
    use crate::gui::inputs::InputOrder;
    use crate::gui::sizing::SizingState;
    use crate::gui::spreadsheet::{INPUT_COLUMNS, INPUTS_SHEET};
    use crate::gui::test_support::{read_xlsx, xlsx_part};
    use crate::{DesignInputs, compute_all};

    /// The Excel serial date of a Unix time (days since 1899-12-30).
    fn serial(unix_s: i64) -> f64 {
        unix_s as f64 / 86_400.0 + 25_569.0
    }

    /// Asserts the cell calamine read (`read`, `Data::Empty` past the end of a row) is `cell`.
    fn assert_cell(cell: &Cell, read: &Data, at: &str) {
        match (&cell.value, read) {
            (CellValue::Empty, Data::Empty) => {}
            (CellValue::Number(x), Data::Float(y)) => assert_eq!(x.to_bits(), y.to_bits(), "{at}"),
            (CellValue::Text(text), Data::String(read)) => {
                assert_eq!(&cell_text(text), read, "{at}")
            }
            (CellValue::Time(unix_s), Data::DateTime(time)) => {
                assert!((time.as_f64() - serial(*unix_s)).abs() < 1e-6, "{at}");
            }
            (want, got) => panic!("{at}: {want:?} read back as {got:?}"),
        }
    }

    /// The snapshot of `inputs` and `results` in the workflow order.
    fn snapshot<'a>(inputs: &'a DesignInputs, results: &'a crate::DesignResults) -> Snapshot<'a> {
        Snapshot {
            inputs,
            results,
            sizing: SizingState::default(),
            sizing_status: None,
            input_order: InputOrder::Workflow,
            share_link: "https://example.test/magcoupling/?m=abc".to_owned(),
            exported_unix_s: 1_790_000_000,
        }
    }

    #[test]
    fn the_workbook_reads_back_cell_for_cell() {
        // The default design, one with failing checks (manual magnets) and one with numbers
        // that are not finite (a positive beta without a grade, E20).
        let mut manual = DesignInputs::default();
        manual.coupling.magnets.part_inner.clear();
        manual.coupling.magnets.part_outer.clear();
        let mut non_finite = manual.clone();
        non_finite.temperature.demag.coercivity_source = 0;
        non_finite.temperature.demag.beta_hcj_per_C = 0.0035;
        for inputs in [DesignInputs::default(), manual, non_finite] {
            let results = compute_all(&inputs);
            let shot = snapshot(&inputs, &results);
            let book = sheets(&shot);
            let read = read_xlsx(&spreadsheet_bytes(&shot).expect("written"));
            let names: Vec<&str> = read.iter().map(|(name, _)| name.as_str()).collect();
            let want: Vec<&str> = book.iter().map(|sheet| sheet.name).collect();
            assert_eq!(names, want);
            for (sheet, (_, rows)) in book.iter().zip(&read) {
                assert_eq!(rows.len(), sheet.rows.len(), "{}", sheet.name);
                for (r, (cells, read_row)) in sheet.rows.iter().zip(rows).enumerate() {
                    for (c, read_cell) in read_row.iter().enumerate() {
                        let at = format!("{} row {} column {}", sheet.name, r + 1, c + 1);
                        assert_cell(cells.get(c).unwrap_or(&Cell::empty()), read_cell, &at);
                    }
                    assert!(cells.len() <= read_row.len().max(1), "{}", sheet.name);
                }
            }
        }
    }

    #[test]
    fn the_file_has_frozen_headers_filters_widths_bold_headings_fills_and_no_formula() {
        let mut inputs = DesignInputs::default();
        inputs.coupling.magnets.part_inner.clear();
        inputs.coupling.magnets.part_outer.clear();
        let results = compute_all(&inputs);
        let shot = snapshot(&inputs, &results);
        let bytes = spreadsheet_bytes(&shot).unwrap();
        let styles = xlsx_part(&bytes, "xl/styles.xml");
        assert!(styles.contains("<b/>"), "bold headings");
        // The three badge colours as fills (ARGB): green, amber, red; and the group headings'
        // grey.
        for rgb in ["FF5AC878", "FFFF8F00", "FFFF0000", "FFE7E6E6"] {
            assert!(
                styles.contains(&format!("rgb=\"{rgb}\"")),
                "{rgb} in {styles}"
            );
        }
        for (index, sheet) in sheets(&shot).iter().enumerate() {
            let xml = xlsx_part(&bytes, &format!("xl/worksheets/sheet{}.xml", index + 1));
            // The header row frozen where the sheet asks (not the Summary).
            assert_eq!(
                xml.contains("ySplit=\"1\"") && xml.contains("state=\"frozen\""),
                sheet.frozen_rows == 1,
                "{}: frozen",
                sheet.name
            );
            assert_eq!(
                xml.contains("<pane"),
                sheet.frozen_rows > 0,
                "{}",
                sheet.name
            );
            // The header's filter buttons over every column and row, where the sheet asks.
            let last_column = char::from(b'A' + u8::try_from(sheet.widths.len() - 1).unwrap());
            let filter = format!("<autoFilter ref=\"A1:{last_column}{}\"", sheet.rows.len());
            assert_eq!(xml.contains(&filter), sheet.autofilter, "{}", sheet.name);
            assert_eq!(
                xml.contains("<autoFilter"),
                sheet.autofilter,
                "{}",
                sheet.name
            );
            // Every column's width, set (rust_xlsxwriter writes a run of equal widths as one
            // <col min max>).
            let set: usize = xml
                .split("<col ")
                .skip(1)
                .map(|col| {
                    let attribute = |name: &str| -> usize {
                        let start = col.find(&format!("{name}=\"")).expect(name) + name.len() + 2;
                        col[start..].split('"').next().unwrap().parse().unwrap()
                    };
                    attribute("max") - attribute("min") + 1
                })
                .sum();
            assert_eq!(set, sheet.widths.len(), "{}", sheet.name);
            assert!(
                !xml.contains("<f>") && !xml.contains("<f "),
                "{}: no formula",
                sheet.name
            );
        }
    }

    #[test]
    fn a_text_longer_than_a_cell_holds_is_cut_to_excel_s_limit_saying_how_long_it_was() {
        let fits = "a".repeat(MAX_CELL_CHARS);
        assert_eq!(cell_text(&fits), fits);
        let long = "é".repeat(40_000);
        let cut = cell_text(&long);
        assert_eq!(cut.chars().count(), MAX_CELL_CHARS);
        assert!(
            cut.ends_with(" [cut: 40000 characters]"),
            "{}",
            &cut[cut.len() - 40..]
        );
        assert!(cut.starts_with("éé"));
        // Excel counts UTF-16 code units: 20,000 emoji are 40,000 of them, over the limit though
        // only 20,000 characters. The cut keeps whole characters, so it may stop one unit short.
        let emoji = "\u{1F600}".repeat(20_000);
        let cut_emoji = cell_text(&emoji);
        let units = cut_emoji.encode_utf16().count();
        assert!(
            (MAX_CELL_CHARS - 1..=MAX_CELL_CHARS).contains(&units),
            "{units}"
        );
        let kept = cut_emoji
            .strip_suffix(" [cut: 20000 characters]")
            .expect("the note");
        assert!(kept.chars().all(|c| c == '\u{1F600}'), "whole characters");
        // At the limit a text is as it is; one unit over, it is cut.
        let at_limit = format!("{}\u{1F600}", "a".repeat(MAX_CELL_CHARS - 2));
        assert_eq!(cell_text(&at_limit), at_limit);
        let over = format!("{}\u{1F600}", "a".repeat(MAX_CELL_CHARS - 1));
        let cut_over = cell_text(&over);
        assert!(cut_over.ends_with(" [cut: 32767 characters]"), "{cut_over}");
        assert!(cut_over.encode_utf16().count() <= MAX_CELL_CHARS);
        // A part name that long and the share link (about 2,500 characters, over Excel's 2,080
        // for a hyperlink: decision X-6) both write.
        let mut inputs = DesignInputs::default();
        inputs.coupling.magnets.part_inner = long.clone();
        inputs.coupling.magnets.part_outer = emoji.clone();
        let results = compute_all(&inputs);
        let mut shot = snapshot(&inputs, &results);
        shot.share_link = format!("https://example.test/magcoupling/?m={}", "A".repeat(2_500));
        let read = read_xlsx(&spreadsheet_bytes(&shot).expect("written"));
        let (_, summary) = &read[0];
        assert!(
            summary
                .iter()
                .any(|row| row.get(1) == Some(&Data::String(shot.share_link.clone())))
        );
        let (_, rows) = read.iter().find(|(name, _)| name == INPUTS_SHEET).unwrap();
        let path = INPUT_COLUMNS.iter().position(|c| *c == "Path").unwrap();
        let part = |input: &str| {
            rows.iter()
                .find(|row| row.get(path) == Some(&Data::String(input.into())))
                .unwrap()
        };
        assert_eq!(part("coupling.magnets.part_inner")[1], Data::String(cut));
        assert_eq!(
            part("coupling.magnets.part_outer")[1],
            Data::String(cut_emoji)
        );
    }

    #[test]
    fn the_level_fills_are_the_page_s_badge_colours() {
        assert_eq!(fill(Level::Good), Color::RGB(0x5A_C8_78));
        assert_eq!(fill(Level::Caution), Color::RGB(0xFF_8F_00));
        assert_eq!(fill(Level::Bad), Color::RGB(0xFF_00_00));
    }

    #[test]
    fn the_export_is_stamped_with_the_time_now() {
        // 2023-11-14: any clock this code runs on is past it.
        assert!(now_unix_s() > 1_700_000_000);
        assert!(
            xlsx_bytes(&[], 1_790_000_000).is_ok(),
            "an empty workbook still writes"
        );
    }
}
````

In `magcoupling-rs/src/gui/mod.rs`, replace:

````rust
pub(crate) mod test_support;
pub mod trace;
pub mod typeset;

pub use format::{SIGNIFICANT_DIGITS, format_value, with_unit};
pub use inputs::KEY_DESIGN;
````

with:

````rust
pub(crate) mod test_support;
pub mod trace;
pub mod typeset;
pub mod xlsx;

pub use format::{SIGNIFICANT_DIGITS, format_value, with_unit};
pub use inputs::KEY_DESIGN;
````

In `magcoupling-rs/src/gui/session.rs`, replace:

````rust
        }
    }

    #[test]
    fn a_share_link_of_the_default_design_stays_short() {
        // Decision M41-5 quotes this length (2,462 characters at the time of writing): the
````

with:

````rust
        }
    }

    /// The share link of the default design as commit 49fbd3f wrote it, before the spreadsheet
    /// export's zip switched flate2 to its zlib-rs backend (decision X-9): a link written since
    /// may differ in its bytes, and every link written before must still load.
    const DEFAULT_PAYLOAD_49FBD3F: &str = concat!(
        "lVhLr5w2FP4vrHMRMMO8dm0WrVq1qpRIXVoeMOBcwMSGO3ca5b_3HNsMNo9pmkWuxuc79nk_-BYUQja0",
        "Dy5BQ8tMDF3N2_IlZ4qXbfAh4G039Cq4fAsyWvOrpD0XbUjrrqLkKknHJPkYXF6iMIri5IMP6kRfsYY0",
        "TXA5h6fUpwLz5-AShzv_OCOszYNLFMYzfEHgFxGSl7ylNSLOcwSjpBfy68Bi8ie8mYTRYQuRaEQcHn1A",
        "STuSs4K3HH8CwCeDhVrWk5q1ZV9pveJkfoXF9BXPXlumlIbtwngdduO5vekQ7mb6NIyqQbLcimwlPs1A",
        "Q4SnSXo47I7sZaaw6mgG7jSihnuf2DMFcrKmQw8m4MEZWfRgcCOnQgCQa9p0Crxfi7sxjj145xoqwTf2",
        "rQl9FWAEkevzJHUI-o9nzGhB7e8dc18CMC1Zw9qevJNc-_hBK2gGliKiZaTjLGMYI6c5tb-JkeoKWUie",
        "GZfbyDPHpeQdaaxSOwdfgVWZPj05p18EB8Guou5JxmVWG0RyWEA2njNElUl2A4vvHuev7E4y0faggjVv",
        "6tGmKNo7T9W8pNpSc4-0A0igzYHvJ4_zTrJa0Bykow_pjtNLQOWsWNymaMH6--NC1yNaEwI_lHKdqGpu",
        "hHL9o3q0tSp87W4Qa0QMrg62RIVXmr1yOWbp41RI5gbTeO7XlfE0l_yNkZkGpib5ECcFI53xIxmYWM9y",
        "X4iSUUlYUfCMsza7P4qVR9dpFlxST0ykXMU70WVXQyB7zasHB8bbFgovWoC4VTaeqQexcaN3qGjdmGGe",
        "7Da3bfo6idgOdb2CKyXNGdFvQ7sINhHgr01EQ9sBHrMKOD3gKXBZcZ_CV4vvU45FHd5Aa9V-RG4D_GG5",
        "Dfz_yG04_lPujsr-4bKfT_vk06_BFmz02xrsnVRUNqLlGYSsS1n2n5HUdqLGMuvGt-geHSfVCQqjB5MQ",
        "flCVQHdWh74NAIMRPaFQUKVznzySf07NoItXYlDw4Cpd1YxBToNAqK4HMLUXKqEQBYkjciZ__EWhyO9m",
        "wvq4ZMSdj-u4O2d1Tn7ak2NkgPu59qpnoPxV0d6EVrokQgfIByjMbxyK1SfS6Fvw39pNWc_G4QxnM5zO",
        "dktUzlqFt5Uka3bB5aintDmoEflQD4r8goInUbr2XDMQCWEG2mLHwfksWZdLdSzjUBhJxUDV38hr-Tvo",
        "cdS9tWHAGdKcdjrJhI0BU8l9YlFTbMA5pyZVtiGmK6ZzKkRzfqPQK0rbsnxyx2vRP66PT1uIpvHZazKZ",
        "tEGTovGT40i_ggttxbHRnXokk9RLEljCLSWOtbzpypEEc8COZWOjnQgWv0_8476SDNr_qPU-9q-zZGcE",
        "m2mfDZ3ba9LJ6kjRrTwTclQ-nh6HkbtmeuRY6I79leBY_hhh7Tkkf489vashwizbg8v17mESsBquW3as",
        "ecEIVIXWzLrRlFiWrPsVy_NxnAZikvpkc5yMh3bE74TijmoO-R1NDT8s694RB4k1zNMmxCdHJpGPEW9M",
        "ol0nxC51EOMGkUtqxgjT2S0V5tqxGL_snYulHl9G01aiXmSRj3jMZiPx68DxVXzALGIOo-I5tq8elevn",
        "NpHwvzYlLC52k3BfHdpxFHRDRFW0wFxVIE02BWY0BYst-Jvus_SV1D3NILgsLwUwNHMcT6cwz-qIImpJ",
        "kF1j4myymy6Qa0IcT4-nYNSDbX1M6mg3nr9RKLLuRmGOb-xqkxSO0NcQLT1EBFSxCjzxxkLFMPPsALuG",
        "IJgZYU55fSfqhub7ONbbbXgBspQwL0OIDJK2ZhdLnnFUUExhBTDJacMgtzm9zaWYfONQIO4Qj2rsFC48",
        "Z5CF4RXuI1X2xftcka4hM8FkZjqsEoPM2MIuBme-j4wr78ftp1GZNwJrNdSCnLz-RHSK7p-Ba_7KcHSy",
        "6NNh9wzdQUZjVhjw8Rw_A-NIBBdLdKNhSA_nDYbsSxJZUJyek3UUTGrM2xfPM9TQ3_UKhS9KrtiGrRBW",
        "0AEWZ7MG2r6ywGCY0ObKMatwhkxXUbqYwwQox1q-irpVmG9S4GcBLx1dZMMVjDBZNQWeqrD4eUMRpFx0",
        "3OBr8wKS0BnHXk7s5TnYuzs-bMskWSYaKHg5hBbOD9iGpjLksuiaU8OsEIIs0Iw_64SM989RhQTvamh0",
        "Oh6egLGxmivndpijnCuPp12yDTYtVSP38XkbZ7usBj7D2SJtLzxsAXHQgY8wDDbZmnxOmj1u6AlsNnG0",
        "xQL2dz6nbOpf8RLGDghxDqsQLh6LtJ6wUFwaSnbxwcz58S5di-I53kSP5jgcnzNgY3DVTMY1Yb8dEx6P",
        "8aNmPIQnu_q5fLZTwYcXUAO3vYU0E4LWA8wKfMAPxc9wWkEYRPbbGLMjYcshf5PfUavd9w-B4v_oZfBb",
        "oOXW3fKKy2ngfvuAjRdSj5lP4bgWw2cf--UHSHZicb4EQVeFu2ECU-Zj8fd_AQ",
    );

    #[test]
    fn a_share_link_written_before_the_zlib_rs_backend_still_loads() {
        // A link names every input: if a default changes later, this link keeps the old value,
        // so compare that input with its old value here; never rewrite the link.
        assert_eq!(DEFAULT_PAYLOAD_49FBD3F.len(), 2_462);
        assert_eq!(
            decode_share_payload(DEFAULT_PAYLOAD_49FBD3F),
            Ok(Design::default())
        );
    }

    #[test]
    fn a_share_link_of_the_default_design_stays_short() {
        // Decision M41-5 quotes this length (2,462 characters at the time of writing): the
````

In `linkage-sim-rs/src/gui/state/file_io.rs`, replace:

````rust
        );
    }

    #[test]
    fn decode_invalid_base64_returns_error() {
        let result = decode_mechanism_from_url("!!!not_valid_base64!!!");
````

with:

````rust
        );
    }

    /// A link written at commit 49fbd3f, before the magcoupling panel's spreadsheet export
    /// (rust_xlsxwriter's zip) switched flate2 to its zlib-rs backend: links written since may
    /// differ in their bytes, and every link written before must still decode.
    #[test]
    fn share_url_written_before_the_zlib_rs_backend_still_decodes() {
        let json = r#"{"schema_version":"1.0.0","bodies":{"ground":{"attachment_points":{"A":[0.0,0.0],"B":[0.1,0.0]},"mass":0.0,"cg_local":[0.0,0.0],"izz_cg":0.0}},"joints":{},"drivers":{},"load_cases":[],"forces":[],"mounting_angle":0.0,"linear_drivers":[]}"#;
        let written = "VU7LCsMgEPyXPUtIrrm1vxGCbNUai7pFbQ8J-feutinksDDDPHY2yGoxAeXbpOwowghD13c9CLiRdibDuIFN9Iq6IiwF1RJMLPJJLpYmX2CcOCL4ZgHXxobGdgEBM5uqDMpKTwr9ye7WVSrbHDvbH0crY51cXfUlnlBLhbkumjh3p6QOHHhecdFKjNab3zfvosEk_yXTvH8A";
        assert_eq!(
            decode_mechanism_from_url(written).expect("decode failed"),
            json
        );
    }

    #[test]
    fn decode_invalid_base64_returns_error() {
        let result = decode_mechanism_from_url("!!!not_valid_base64!!!");
````


- [ ] **Step 2: Run the tests to see them fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-xlsx/magcoupling-rs/Cargo.toml --features app --lib xlsx 2>&1 | grep -E "^error" | sort | uniq -c | sort -rn
```

Expected (cargo first locks the new crates on its stderr, for information only: on 2026-10-03 there were 13, `Adding calamine v0.36.1`, `rust_xlsxwriter v0.99.1`, `zip v8.6.0` and `zopfli v0.8.3` among them; a different list or version is fine except for rust_xlsxwriter, which must be `v0.99.1`; the grep keeps only the errors): FAIL to compile, with these lines (the module's imports are not written yet):

```
      7 error[E0425]: cannot find value `MAX_CELL_CHARS` in this scope
      6 error[E0425]: cannot find function `cell_text` in this scope
      4 error[E0433]: failed to resolve: use of undeclared type `CellValue`
      3 error[E0433]: failed to resolve: use of undeclared type `Level`
      3 error[E0433]: failed to resolve: use of undeclared type `Color`
      3 error[E0425]: cannot find function `spreadsheet_bytes` in this scope
      3 error[E0425]: cannot find function `fill` in this scope
      2 error[E0425]: cannot find function `sheets` in this scope
      1 error[E0433]: failed to resolve: use of undeclared type `Cell`
      1 error[E0425]: cannot find function `xlsx_bytes` in this scope
      1 error[E0425]: cannot find function `now_unix_s` in this scope
      1 error[E0422]: cannot find struct, variant or union type `Snapshot` in this scope
      1 error[E0412]: cannot find type `Snapshot` in this scope
      1 error[E0412]: cannot find type `Cell` in this scope
      1 error: could not compile `magcoupling-rs` (lib test) due to 37 previous errors; 1 warning emitted
```


- [ ] **Step 3: Write the writer, extend gate 11, document the dependency**

In `magcoupling-rs/src/gui/xlsx.rs`, replace:

````rust
#[cfg(test)]
mod tests {
````

with:

````rust
//! Writes the spreadsheet export ([`crate::gui::spreadsheet`]) as an .xlsx workbook with
//! rust_xlsxwriter (decision X-9): a number as a number cell, a text as a text cell (cut to
//! Excel's 32,767 characters, as Excel counts them, ending with how long it was), a time as a
//! date, each sheet's header rows frozen, its header's filter buttons and its columns' widths as
//! the sheet asks, the headings bold (a group heading on a light grey), and a check's level
//! filled with the badge's colour on the page (decision X-11). Values only: no cell holds a
//! formula (decision X-1).

use rust_xlsxwriter::{
    Color, DocProperties, ExcelDateTime, Format, FormatBorder, Workbook, Worksheet, XlsxError,
};

use crate::gui::dashboard::Level;
use crate::gui::panel::HEADING;
use crate::gui::spreadsheet::{Cell, CellStyle, CellValue, Sheet, Snapshot, sheets};

/// The file name the export suggests (decision X-7): the stem of the CSV and JSON exports.
pub const XLSX_FILE_NAME: &str = "magcoupling-results.xlsx";

/// The media type of an .xlsx file (the web download's Blob type).
pub const XLSX_MIME: &str = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet";

/// The most characters an Excel cell holds, as Excel counts them: UTF-16 code units (a character
/// outside the Basic Multilingual Plane, an emoji, counts two).
pub const MAX_CELL_CHARS: usize = 32_767;

/// The fill of a group heading (an input group, a result chain, "Other results", a Summary
/// block): a light grey that sets it apart from the section headings inside it.
pub const GROUP_FILL: Color = Color::RGB(0xE7_E6_E6);

/// The number format of a time cell.
pub const TIME_FORMAT: &str = "yyyy-mm-dd hh:mm:ss";

/// The spreadsheet of `snapshot` as the bytes of an .xlsx file, or why it could not be written.
pub fn spreadsheet_bytes(snapshot: &Snapshot) -> Result<Vec<u8>, String> {
    xlsx_bytes(&sheets(snapshot), snapshot.exported_unix_s).map_err(|error| error.to_string())
}

/// `sheets` as the bytes of an .xlsx file whose document properties say it was created at
/// `created_unix_s` [s since the Unix epoch].
pub fn xlsx_bytes(sheets: &[Sheet], created_unix_s: i64) -> Result<Vec<u8>, XlsxError> {
    let mut workbook = Workbook::new();
    let created = ExcelDateTime::from_timestamp(created_unix_s)?;
    workbook.set_properties(
        &DocProperties::new()
            .set_title(HEADING)
            .set_creation_datetime(&created),
    );
    for sheet in sheets {
        let worksheet = workbook.add_worksheet();
        worksheet.set_name(sheet.name)?;
        for (col, width) in sheet.widths.iter().enumerate() {
            worksheet.set_column_width(column(col)?, *width)?;
        }
        if sheet.frozen_rows > 0 {
            worksheet.set_freeze_panes(sheet.frozen_rows, 0)?;
        }
        if sheet.autofilter && !sheet.rows.is_empty() && !sheet.widths.is_empty() {
            let last_row =
                u32::try_from(sheet.rows.len() - 1).map_err(|_| XlsxError::RowColumnLimitError)?;
            worksheet.autofilter(0, 0, last_row, column(sheet.widths.len() - 1)?)?;
        }
        for (row, cells) in sheet.rows.iter().enumerate() {
            let row = u32::try_from(row).map_err(|_| XlsxError::RowColumnLimitError)?;
            for (col, cell) in cells.iter().enumerate() {
                write_cell(worksheet, row, column(col)?, cell)?;
            }
        }
    }
    workbook.save_to_buffer()
}

/// A column index as rust_xlsxwriter takes it.
fn column(col: usize) -> Result<u16, XlsxError> {
    u16::try_from(col).map_err(|_| XlsxError::RowColumnLimitError)
}

/// Writes `cell` at (`row`, `col`); an empty cell is not written.
fn write_cell(worksheet: &mut Worksheet, row: u32, col: u16, cell: &Cell) -> Result<(), XlsxError> {
    let format = format_of(cell.style);
    match &cell.value {
        CellValue::Empty => {}
        CellValue::Number(x) => {
            worksheet.write_number_with_format(row, col, *x, &format)?;
        }
        CellValue::Text(text) => {
            worksheet.write_string_with_format(row, col, cell_text(text), &format)?;
        }
        CellValue::Time(unix_s) => {
            let time = ExcelDateTime::from_timestamp(*unix_s)?;
            let format = format.set_num_format(TIME_FORMAT);
            worksheet.write_datetime_with_format(row, col, &time, &format)?;
        }
    }
    Ok(())
}

/// The format of a cell style: the title, headers and headings bold (the column headers
/// underlined, a group heading on [`GROUP_FILL`]), a check's level bold on the badge's colour
/// ([`fill`]).
fn format_of(style: CellStyle) -> Format {
    match style {
        CellStyle::Plain => Format::new(),
        CellStyle::Title => Format::new().set_bold().set_font_size(14),
        CellStyle::Header => Format::new()
            .set_bold()
            .set_border_bottom(FormatBorder::Thin),
        CellStyle::Group => Format::new()
            .set_bold()
            .set_font_size(12)
            .set_background_color(GROUP_FILL),
        CellStyle::Section => Format::new().set_bold(),
        CellStyle::Level(level) => Format::new().set_bold().set_background_color(fill(level)),
    }
}

/// The badge colour of `level` on the page, which shows egui's dark visuals (decision X-11):
/// green, the warning amber, the error red. Black text reads on each.
pub fn fill(level: Level) -> Color {
    let color = level.color(&egui::Visuals::dark());
    Color::RGB((u32::from(color.r()) << 16) | (u32::from(color.g()) << 8) | u32::from(color.b()))
}

/// `text` as a cell holds it: as it is while it fits [`MAX_CELL_CHARS`] UTF-16 code units (as
/// Excel counts); a longer text cut after the last whole character that leaves room for a note
/// of how many characters it had.
pub fn cell_text(text: &str) -> String {
    if text.encode_utf16().count() <= MAX_CELL_CHARS {
        return text.to_owned();
    }
    let note = format!(" [cut: {} characters]", text.chars().count());
    let mut room = MAX_CELL_CHARS - note.encode_utf16().count();
    let mut cut = String::new();
    for c in text.chars() {
        if c.len_utf16() > room {
            break;
        }
        room -= c.len_utf16();
        cut.push(c);
    }
    cut.push_str(&note);
    cut
}

/// Now [s since the Unix epoch]: the time an export is stamped with (decision X-12).
pub fn now_unix_s() -> i64 {
    #[cfg(target_arch = "wasm32")]
    {
        (js_sys::Date::now() / 1000.0) as i64
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |elapsed| i64::try_from(elapsed.as_secs()).unwrap_or(0))
    }
}

#[cfg(test)]
mod tests {
````

In `linkage-sim-rs/scripts/gate.sh`, replace:

````bash
# and wasm32, the guard that no shipped build (the calculator's own and the
# linkage app's, native and wasm32) has the test-only workbook-parity feature
# (with negative controls that prove the guard trips), and the check that both
# crates lock the same egui, egui_plot, eframe and wasm-bindgen (the CLI
# version deploy-web.yml installs).
# Gate 12: the vendored Python oracle, reference/magcoupling-py: its parity
# suite, and a check that the committed differential test data is current.
# Gate 12 needs a Python with the oracle's dependencies; see oracle_python
````

with:

````bash
# and wasm32, the guard that no shipped build (the calculator's own and the
# linkage app's, native and wasm32) has the test-only workbook-parity feature
# (with negative controls that prove the guard trips), and the check that both
# crates lock the same egui, egui_plot, eframe, rust_xlsxwriter and wasm-bindgen
# (the CLI version deploy-web.yml installs).
# Gate 12: the vendored Python oracle, reference/magcoupling-py: its parity
# suite, and a check that the committed differential test data is current.
# Gate 12 needs a Python with the oracle's dependencies; see oracle_python
````

In `linkage-sim-rs/scripts/gate.sh`, replace:

````bash
assert_guard_trips linkage-gui magcoupling-rs/workbook-parity "${LINKAGE_NATIVE_ARGS[@]}"
assert_guard_trips linkage-web magcoupling-rs/workbook-parity "${LINKAGE_WEB_ARGS[@]}"

echo "== gate 11/12: lock parity (egui, egui_plot, eframe, wasm-bindgen; wasm-bindgen-cli pin in deploy-web.yml) =="
LINKAGE_LOCK="Cargo.lock"
MAGCOUPLING_LOCK="$REPO_ROOT/magcoupling-rs/Cargo.lock"
for pkg in egui egui_plot eframe wasm-bindgen; do
  linkage_version="$(lock_versions "$LINKAGE_LOCK" "$pkg")"
  magcoupling_version="$(lock_versions "$MAGCOUPLING_LOCK" "$pkg")"
  if [[ -z "$linkage_version" || "$linkage_version" != "$magcoupling_version" ]]; then
````

with:

````bash
assert_guard_trips linkage-gui magcoupling-rs/workbook-parity "${LINKAGE_NATIVE_ARGS[@]}"
assert_guard_trips linkage-web magcoupling-rs/workbook-parity "${LINKAGE_WEB_ARGS[@]}"

echo "== gate 11/12: lock parity (egui, egui_plot, eframe, rust_xlsxwriter, wasm-bindgen; wasm-bindgen-cli pin in deploy-web.yml) =="
LINKAGE_LOCK="Cargo.lock"
MAGCOUPLING_LOCK="$REPO_ROOT/magcoupling-rs/Cargo.lock"
for pkg in egui egui_plot eframe rust_xlsxwriter wasm-bindgen; do
  linkage_version="$(lock_versions "$LINKAGE_LOCK" "$pkg")"
  magcoupling_version="$(lock_versions "$MAGCOUPLING_LOCK" "$pkg")"
  if [[ -z "$linkage_version" || "$linkage_version" != "$magcoupling_version" ]]; then
````

In `magcoupling-rs/README.md`, replace (part of one line):

````markdown
| egui 0.32, egui_plot 0.33, serde_json, base64, flate2, log |
````

with:

````markdown
| egui 0.32, egui_plot 0.33, serde_json, base64, flate2, log, rust_xlsxwriter 0.99.1 (the spreadsheet export; its `wasm` feature and js-sys on wasm32) |
````


- [ ] **Step 4: Run the tests to see them pass**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-xlsx/magcoupling-rs/Cargo.toml --features app --lib xlsx 2>&1 | grep -E "^test |test result"
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-xlsx/magcoupling-rs/Cargo.toml --features app --lib a_share_link_written_before 2>&1 | grep -E "^test |test result"
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-xlsx/magcoupling-rs/Cargo.toml --features app --lib 2>&1 | grep -E "test result"
```

Expected: the five `gui::xlsx::tests` `ok` and `test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 513 filtered out`; `gui::session::tests::a_share_link_written_before_the_zlib_rs_backend_still_loads ... ok` and `test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 517 filtered out`; the library `test result: ok. 518 passed`.

- [ ] **Step 5: Update the linkage lock file and check both locks and the wasm32 feature**

Run:

```bash
cargo check --manifest-path C:/Users/Cole/source/repos/lsim-mag-xlsx/linkage-sim-rs/Cargo.toml 2>&1 | grep -E "Adding|Updating|Removing|error"
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-xlsx/linkage-sim-rs/Cargo.toml --lib share_url_written_before 2>&1 | grep -E "^test |test result"
for lock in magcoupling-rs linkage-sim-rs; do for pkg in rust_xlsxwriter wasm-bindgen zip; do printf '%s %s ' $lock $pkg; grep -A1 "^name = \"$pkg\"$" C:/Users/Cole/source/repos/lsim-mag-xlsx/$lock/Cargo.lock | sed -n 2p; done; done
cargo tree --manifest-path C:/Users/Cole/source/repos/lsim-mag-xlsx/magcoupling-rs/Cargo.toml --features gui --target wasm32-unknown-unknown -e features -i rust_xlsxwriter 2>/dev/null | grep -c 'rust_xlsxwriter feature "wasm"'
cargo tree --manifest-path C:/Users/Cole/source/repos/lsim-mag-xlsx/magcoupling-rs/Cargo.toml --features gui -e features -i rust_xlsxwriter 2>/dev/null | grep -c 'rust_xlsxwriter feature "wasm"'
cargo clippy --manifest-path C:/Users/Cole/source/repos/lsim-mag-xlsx/magcoupling-rs/Cargo.toml --target wasm32-unknown-unknown --features gui --lib -- -D warnings 2>&1 | tail -n 1
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-xlsx/magcoupling-rs/Cargo.toml --check
```

Expected: cargo updates the index and adds crates to the linkage lock (for information, as in Step 2), the linkage app's pinned link still decodes, then the version lines, the two counts (the `wasm` feature on for wasm32: any count above 0; off natively: 0), clippy's last line, and nothing from `fmt --check`:

```
    Updating crates.io index
      Adding rust_xlsxwriter v0.99.1
      Adding typed-path v0.12.3
      Adding zip v8.6.0
      Adding zlib-rs v0.6.8
      Adding zopfli v0.8.3
test gui::state::file_io::tests::share_url_written_before_the_zlib_rs_backend_still_decodes ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 956 filtered out; finished in ...
magcoupling-rs rust_xlsxwriter version = "0.99.1"
magcoupling-rs wasm-bindgen version = "0.2.114"
magcoupling-rs zip version = "8.6.0"
linkage-sim-rs rust_xlsxwriter version = "0.99.1"
linkage-sim-rs wasm-bindgen version = "0.2.114"
linkage-sim-rs zip version = "8.6.0"
3
0
    Finished `dev` profile [unoptimized + debuginfo] target(s) in ...
```

If either lock does not hold `rust_xlsxwriter` `0.99.1` and `wasm-bindgen` `0.2.114`, stop and escalate (gate 11 and deploy-web.yml's wasm-bindgen-cli pin); never run `cargo update`. Other `Adding` lines or another zip version are not a reason to stop (cargo picks the newest release compatible with Rust 1.89: in Step 2 it says `Locking 13 packages to latest Rust 1.89.0 compatible versions`): record them for the review.

- [ ] **Step 6: Gate**

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-xlsx/linkage-sim-rs/scripts/gate.sh 2>&1 | grep -E "lock files|deploy-web|GATE|SKIP|FAIL"
git -C C:/Users/Cole/source/repos/lsim-mag-xlsx checkout -- docs/chebyshev_lambda
```

Expected: gate 11's heading line (it names deploy-web.yml), then `egui 0.32.3 in both lock files`, `egui_plot 0.33.0 in both lock files`, `eframe 0.32.3 in both lock files`, `rust_xlsxwriter 0.99.1 in both lock files`, `wasm-bindgen 0.2.114 in both lock files`, `deploy-web.yml installs wasm-bindgen-cli 0.2.114`, `GATE PASS`; no `SKIP gate` or `FAIL` line.

- [ ] **Step 7: Commit**

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-xlsx add magcoupling-rs/Cargo.toml magcoupling-rs/Cargo.lock linkage-sim-rs/Cargo.lock magcoupling-rs/src/gui/test_support.rs magcoupling-rs/src/gui/xlsx.rs magcoupling-rs/src/gui/mod.rs magcoupling-rs/src/gui/session.rs linkage-sim-rs/src/gui/state/file_io.rs linkage-sim-rs/scripts/gate.sh magcoupling-rs/README.md
git -C C:/Users/Cole/source/repos/lsim-mag-xlsx commit -m "feat(magcoupling-rs): write the spreadsheet as .xlsx with rust_xlsxwriter 0.99.1 (decisions X-7, X-9, X-11)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9"
git -C C:/Users/Cole/source/repos/lsim-mag-xlsx status --short
```

Expected: one commit; no status lines.

### Task 4: The buttons, the linkage save filter, the docs and the browser check

**Model:** `sonnet` (exact code and text; CLAUDE.md section 5). Escalate a retry to the session model.

**Files:**
- Modify: `magcoupling-rs/src/gui/panel.rs` (`EXPORT_SPREADSHEET` in the header; `export_spreadsheet`; the `ExportXlsx` arm; two tests and their helpers)
- Modify: `magcoupling-rs/src/gui/results_table.rs` (`EXPORT_XLSX`, `TableAction::ExportXlsx`, its button)
- Modify: `linkage-sim-rs/src/gui/calculator_window.rs` (the Excel workbook save filter; its test row)
- Modify: `magcoupling-rs/README.md`, `docs/ai/02-system.yaml`, `docs/ai/03-structure.yaml`, `docs/ai/04-memory.yaml`, `docs/ai/05-update-tracker.md`
- Create: `docs/superpowers/plans/2026-10-03-magcoupling-xlsx-export.md` (this plan, copied)

**Interfaces:**
- Consumes: `Snapshot` (Task 2); `spreadsheet_bytes`, `now_unix_s`, `XLSX_FILE_NAME`, `XLSX_MIME` (Task 3); `read_xlsx` (Task 3's test helper); `MagcouplingPanel::{shown_inputs, share_link, report}`, `SizingRunner::status`.
- Produces: `pub const EXPORT_SPREADSHEET: &str = "Export spreadsheet"` (panel); `pub const EXPORT_XLSX: &str = "Export XLSX"` and `TableAction::ExportXlsx` (results_table); private `MagcouplingPanel::export_spreadsheet(&mut self)` (queues `PanelRequest::SaveFile { file_name: XLSX_FILE_NAME, mime: XLSX_MIME, contents }`, or reports "Could not write the spreadsheet: ..." in the header); the linkage `save_filter` gives `.xlsx` the filter `Excel workbook` (`["xlsx"]`).

- [ ] **Step 1: Write the failing tests**

In `magcoupling-rs/src/gui/panel.rs`, replace:

````rust
            ]
        );
        assert!(harness.panel.take_requests().is_empty(), "drained");
    }

    #[test]
````

with:

````rust
            ]
        );
        assert!(harness.panel.take_requests().is_empty(), "drained");
    }

    /// The spreadsheets the panel queued since the last call, each read back sheet by sheet;
    /// every request must be a spreadsheet's.
    fn queued_spreadsheets(harness: &mut Harness) -> Vec<Vec<(String, Vec<Vec<calamine::Data>>)>> {
        use crate::gui::test_support::read_xlsx;
        use crate::gui::xlsx::{XLSX_FILE_NAME, XLSX_MIME};
        harness
            .panel
            .take_requests()
            .into_iter()
            .map(|request| match request {
                PanelRequest::SaveFile {
                    file_name,
                    mime,
                    contents,
                } => {
                    assert_eq!(file_name, XLSX_FILE_NAME);
                    assert_eq!(mime, XLSX_MIME);
                    read_xlsx(&contents)
                }
                PanelRequest::OpenDesign => panic!("not a spreadsheet"),
            })
            .collect()
    }

    /// The row of a read-back sheet whose column `column` holds the text `wanted`.
    fn read_row<'r>(
        rows: &'r [Vec<calamine::Data>],
        column: usize,
        wanted: &str,
    ) -> &'r Vec<calamine::Data> {
        rows.iter()
            .find(|row| row.get(column) == Some(&calamine::Data::String(wanted.to_owned())))
            .unwrap_or_else(|| panic!("no row {wanted}"))
    }

    #[test]
    fn both_spreadsheet_buttons_queue_the_workbook_of_the_design_shown() {
        use crate::gui::results_table::EXPORT_XLSX;
        use crate::gui::spreadsheet::{
            ASSUMPTIONS_SHEET, INPUT_COLUMNS, INPUTS_SHEET, RESULTS_SHEET, SHARE_LINK,
            SUMMARY_SHEET,
        };
        let mut harness = Harness::new();
        assert!(harness.panel.take_requests().is_empty());
        harness.focus(FACE_GAP);
        harness.frame(key_tap(egui::Key::ArrowRight));
        // The header's button, then the results table's.
        harness.click_text(EXPORT_SPREADSHEET);
        harness.click_text(CentreView::Results.label());
        harness.click_text(EXPORT_XLSX);
        let books = queued_spreadsheets(&mut harness);
        assert_eq!(books.len(), 2);
        let path = INPUT_COLUMNS.iter().position(|c| *c == "Path").unwrap();
        for book in &books {
            let names: Vec<&str> = book.iter().map(|(name, _)| name.as_str()).collect();
            assert_eq!(
                names,
                [
                    SUMMARY_SHEET,
                    INPUTS_SHEET,
                    RESULTS_SHEET,
                    ASSUMPTIONS_SHEET
                ]
            );
            // The face gap as edited, a number; the link that reopens this design.
            let gap = read_row(&book[1].1, path, FACE_GAP);
            assert_eq!(gap[1], calamine::Data::Float(1.41));
            let link = read_row(&book[0].1, 0, SHARE_LINK);
            assert_eq!(link[1], calamine::Data::String(harness.panel.share_link()));
        }
        assert!(harness.panel.take_requests().is_empty(), "drained");
    }

    #[test]
    fn the_spreadsheet_follows_the_input_order_shown_and_torque_to_magnets() {
        use crate::gui::spreadsheet::{INPUT_COLUMNS, SIZING_MODE, SIZING_OUTCOME};
        let column = |name: &str| INPUT_COLUMNS.iter().position(|c| *c == name).unwrap();
        let mut harness = Harness::new();
        // The workbook order: the Inputs sheet's first heading is the first package group's.
        harness.click_text(InputOrder::Workbook.label());
        harness.click_text(EXPORT_SPREADSHEET);
        let books = queued_spreadsheets(&mut harness);
        let first_group = InputCatalogue::get().groups[0].label;
        assert_eq!(
            books[0][1].1[1][0],
            calamine::Data::String(first_group.to_owned())
        );
        // Torque -> Magnets: the solved length, noted, and the sizing on the Summary.
        harness.size();
        harness.click_text(EXPORT_SPREADSHEET);
        let books = queued_spreadsheets(&mut harness);
        let (summary, inputs) = (&books[0][0].1, &books[0][1].1);
        assert_eq!(
            read_row(summary, 0, SIZING_MODE)[1],
            calamine::Data::String(SizingMode::TorqueToMagnets.label().to_owned())
        );
        let outcome = read_row(summary, 0, SIZING_OUTCOME);
        assert!(
            matches!(&outcome[1], calamine::Data::String(text) if text.starts_with(SOLVED_PREFIX)),
            "{:?}",
            outcome[1]
        );
        let length = read_row(inputs, column("Path"), AXIAL_LENGTH);
        let shown = harness
            .panel
            .shown_inputs()
            .coupling
            .magnets
            .axial_length_mm;
        assert_eq!(length[1], calamine::Data::Float(shown.expect("sized")));
        assert_eq!(
            length[column("Note")],
            calamine::Data::String(SIZED_NOTE.to_owned())
        );
    }

    #[test]
````

In `linkage-sim-rs/src/gui/calculator_window.rs`, replace:

````rust
    };
    use magcoupling::gui::results_table::{CSV_FILE_NAME, JSON_FILE_NAME};
    use magcoupling::gui::session::{Design, PUBLIC_BASE_URL, design_to_json};

    /// The panel's heading (magcoupling-rs `gui::panel::HEADING`), a label: a click on it does
    /// nothing but give the window the keyboard.
````

with:

````rust
    };
    use magcoupling::gui::results_table::{CSV_FILE_NAME, JSON_FILE_NAME};
    use magcoupling::gui::session::{Design, PUBLIC_BASE_URL, design_to_json};
    use magcoupling::gui::xlsx::XLSX_FILE_NAME;

    /// The panel's heading (magcoupling-rs `gui::panel::HEADING`), a label: a click on it does
    /// nothing but give the window the keyboard.
````

In `linkage-sim-rs/src/gui/calculator_window.rs`, replace:

````rust
            (JSON_FILE_NAME, "JSON", "json"),
            ("magcoupling-design.json", "JSON", "json"),
            (CSV_FILE_NAME, "CSV", "csv"),
            ("notes.txt", "All files", "*"),
            ("no_extension", "All files", "*"),
        ] {
````

with:

````rust
            (JSON_FILE_NAME, "JSON", "json"),
            ("magcoupling-design.json", "JSON", "json"),
            (CSV_FILE_NAME, "CSV", "csv"),
            (XLSX_FILE_NAME, "Excel workbook", "xlsx"),
            ("notes.txt", "All files", "*"),
            ("no_extension", "All files", "*"),
        ] {
````


- [ ] **Step 2: Run the tests to see them fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-xlsx/magcoupling-rs/Cargo.toml --features app --lib spreadsheet 2>&1 | grep -E "^error" | sort | uniq -c
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-xlsx/linkage-sim-rs/Cargo.toml --lib saved_files_get_the_filter_of_their_type 2>&1 | grep -E "panicked|left|right|test result"
```

Expected: magcoupling-rs fails to compile its tests:

```
      1 error: could not compile `magcoupling-rs` (lib test) due to 4 previous errors
      3 error[E0425]: cannot find value `EXPORT_SPREADSHEET` in this scope
      1 error[E0432]: unresolved import `crate::gui::results_table::EXPORT_XLSX`
```

and linkage-sim-rs compiles, but its filter test fails:

```
thread 'gui::calculator_window::tests::saved_files_get_the_filter_of_their_type' panicked at src\gui\calculator_window.rs:1025:13:
assertion `left == right` failed: magcoupling-results.xlsx
  left: ("All files", ["*"])
 right: ("Excel workbook", ["xlsx"])
test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 956 filtered out; finished in 0.00s
```


- [ ] **Step 3: Add the buttons and the filter**

In `magcoupling-rs/src/gui/panel.rs`, replace:

````rust
    SOLVED_PREFIX, SOLVING, SizingMode, SizingRunner, SizingState, TARGET_RANGE_INPUT,
    variable_label,
};
use crate::gui::trace::{CLEAR_TRACE, Trace, TraceKind};
use crate::{DesignInputs, DesignResults, compute_all};

/// The heading of the panel.
````

with:

````rust
    SOLVED_PREFIX, SOLVING, SizingMode, SizingRunner, SizingState, TARGET_RANGE_INPUT,
    variable_label,
};
use crate::gui::spreadsheet::Snapshot;
use crate::gui::trace::{CLEAR_TRACE, Trace, TraceKind};
use crate::gui::xlsx::{XLSX_FILE_NAME, XLSX_MIME, now_unix_s, spreadsheet_bytes};
use crate::{DesignInputs, DesignResults, compute_all};

/// The heading of the panel.
````

In `magcoupling-rs/src/gui/panel.rs`, replace:

````rust
pub const SAVE_DESIGN: &str = "Save design";
pub const LOAD_DESIGN: &str = "Load design";
pub const COPY_SHARE_LINK: &str = "Copy share link";

/// The file name a saved design suggests.
pub const DESIGN_FILE_NAME: &str = "magcoupling-design.json";
````

with:

````rust
pub const SAVE_DESIGN: &str = "Save design";
pub const LOAD_DESIGN: &str = "Load design";
pub const COPY_SHARE_LINK: &str = "Copy share link";

/// The header's spreadsheet button (decision X-8; the results table has its own, beside the CSV
/// and JSON exports).
pub const EXPORT_SPREADSHEET: &str = "Export spreadsheet";

/// The file name a saved design suggests.
pub const DESIGN_FILE_NAME: &str = "magcoupling-design.json";
````

In `magcoupling-rs/src/gui/panel.rs`, replace:

````rust
                        )
                        .into_bytes(),
                    }),
                    None => {}
                }
            }
        }
    }
````

with:

````rust
                        )
                        .into_bytes(),
                    }),
                    Some(TableAction::ExportXlsx) => self.export_spreadsheet(),
                    None => {}
                }
            }
        }
    }

    /// Queues the spreadsheet of the design shown (decision X-8: both spreadsheet buttons) for
    /// the host to save, stamped with the time now; when it cannot be written, the header says
    /// why instead.
    fn export_spreadsheet(&mut self) {
        let shown = self.shown_inputs();
        let snapshot = Snapshot {
            inputs: &shown,
            results: &self.results,
            sizing: self.sizing,
            sizing_status: (self.sizing.mode == SizingMode::TorqueToMagnets)
                .then(|| self.runner.status(&self.inputs, &self.sizing)),
            input_order: self.input_order,
            share_link: self.share_link(),
            exported_unix_s: now_unix_s(),
        };
        match spreadsheet_bytes(&snapshot) {
            Ok(contents) => self.requests.push(PanelRequest::SaveFile {
                file_name: XLSX_FILE_NAME.to_owned(),
                mime: XLSX_MIME,
                contents,
            }),
            Err(error) => self.report(Err(format!("Could not write the spreadsheet: {error}"))),
        }
    }
````

In `magcoupling-rs/src/gui/panel.rs`, replace:

````rust
                    link.len()
                ));
                ui.ctx().copy_text(link);
            }
            ui.separator();
            if ui
````

with:

````rust
                    link.len()
                ));
                ui.ctx().copy_text(link);
            }
            if ui
                .button(EXPORT_SPREADSHEET)
                .on_hover_text(
                    "This design and its results as an .xlsx workbook laid out as this page (values, no formulas)",
                )
                .clicked()
            {
                self.export_spreadsheet();
            }
            ui.separator();
            if ui
````

In `magcoupling-rs/src/gui/results_table.rs`, replace:

````rust
/// The button labels.
pub const EXPORT_CSV: &str = "Export CSV";
pub const EXPORT_JSON: &str = "Export JSON";

/// The search box's hint.
pub const SEARCH_HINT: &str = "Search label, path or cell";
````

with:

````rust
/// The button labels.
pub const EXPORT_CSV: &str = "Export CSV";
pub const EXPORT_JSON: &str = "Export JSON";
/// The spreadsheet (decision X-8): the page's layout as an .xlsx workbook (`gui::xlsx`).
pub const EXPORT_XLSX: &str = "Export XLSX";

/// The search box's hint.
pub const SEARCH_HINT: &str = "Search label, path or cell";
````

In `magcoupling-rs/src/gui/results_table.rs`, replace:

````rust
pub enum TableAction {
    ExportCsv,
    ExportJson,
}

impl ResultsTable {
````

with:

````rust
pub enum TableAction {
    ExportCsv,
    ExportJson,
    ExportXlsx,
}

impl ResultsTable {
````

In `magcoupling-rs/src/gui/results_table.rs`, replace:

````rust
            }
            if ui.button(EXPORT_JSON).clicked() {
                action = Some(TableAction::ExportJson);
            }
        });
        ui.separator();
````

with:

````rust
            }
            if ui.button(EXPORT_JSON).clicked() {
                action = Some(TableAction::ExportJson);
            }
            if ui
                .button(EXPORT_XLSX)
                .on_hover_text("The page's layout as an .xlsx workbook (values, no formulas)")
                .clicked()
            {
                action = Some(TableAction::ExportXlsx);
            }
        });
        ui.separator();
````

In `linkage-sim-rs/src/gui/calculator_window.rs`, replace:

````rust
}

/// The native save dialog's filter for a file the panel saves: its design file and results
/// exports are JSON or CSV.
fn save_filter(file_name: &str) -> FileFilter {
    match file_name.rsplit_once('.').map(|(_, extension)| extension) {
        Some("json") => FileFilter {
````

with:

````rust
}

/// The native save dialog's filter for a file the panel saves: its design file and results
/// exports are JSON or CSV, its spreadsheet an .xlsx workbook.
fn save_filter(file_name: &str) -> FileFilter {
    match file_name.rsplit_once('.').map(|(_, extension)| extension) {
        Some("json") => FileFilter {
````

In `linkage-sim-rs/src/gui/calculator_window.rs`, replace:

````rust
        Some("csv") => FileFilter {
            label: "CSV",
            extensions: &["csv"],
        },
        _ => FileFilter {
            label: "All files",
````

with:

````rust
        Some("csv") => FileFilter {
            label: "CSV",
            extensions: &["csv"],
        },
        Some("xlsx") => FileFilter {
            label: "Excel workbook",
            extensions: &["xlsx"],
        },
        _ => FileFilter {
            label: "All files",
````


- [ ] **Step 4: Run the tests to see them pass**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-xlsx/magcoupling-rs/Cargo.toml --features app --lib spreadsheet 2>&1 | grep -E "^test |test result"
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-xlsx/linkage-sim-rs/Cargo.toml --lib saved_files_get_the_filter_of_their_type 2>&1 | grep -E "test result"
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-xlsx/magcoupling-rs/Cargo.toml --features app 2>&1 | grep -E "test result" | head -n 1
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-xlsx/linkage-sim-rs/Cargo.toml --lib 2>&1 | grep -E "test result"
git -C C:/Users/Cole/source/repos/lsim-mag-xlsx checkout -- docs/chebyshev_lambda
```

Expected: the 12 spreadsheet tests and the panel's `both_spreadsheet_buttons_queue_the_workbook_of_the_design_shown` and `the_spreadsheet_follows_the_input_order_shown_and_torque_to_magnets` `ok` and `test result: ok. 14 passed; 0 failed; 0 ignored; 0 measured; 506 filtered out`; the filter test `test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 956 filtered out`; the magcoupling library `test result: ok. 520 passed` (the panel's layout tests among them: with the header's new button, `the_panel_works_inside_an_egui_window` still keeps the linkage window's 1100 x 700); the linkage library `test result: ok. 957 passed`.

- [ ] **Step 5: Update the docs and commit this plan with them**

In `magcoupling-rs/README.md`, replace:

````markdown
  value shown) at full precision, a non-finite number written `+inf`, `-inf` or `NaN` (M41-15).
  The label column flexes (`column_widths`, M42-8: whatever the value, cell and marker columns
  leave, at least 120 points), so a ~930 px window shows each row's label and value without
  scrolling.
````

with:

````markdown
  value shown) at full precision, a non-finite number written `+inf`, `-inf` or `NaN` (M41-15);
  "Export XLSX" saves the spreadsheet (below).
  The label column flexes (`column_widths`, M42-8: whatever the value, cell and marker columns
  leave, at least 120 points), so a ~930 px window shows each row's label and value without
  scrolling.
- **Spreadsheet export** (`spreadsheet.rs`, `xlsx.rs`; plan
  `docs/superpowers/plans/2026-10-03-magcoupling-xlsx-export.md`, decisions X-1 to X-12): "Export
  spreadsheet" in the header and "Export XLSX" beside the CSV and JSON exports save the design
  shown as `magcoupling-results.xlsx`, laid out as the page. Sheets: **Summary** (the title, the
  UTC export time, the app version, the share link as text (about 2,500 characters, over Excel's
  2,080-character hyperlink limit: paste it into a browser), the sizing state, the end-effect and
  assumptions banners, the dashboard's rows with their badges, the material warnings and the
  corrections applied); **Inputs** in the order the inputs side shows (workflow or workbook
  groups, with the section and Advanced headings, drawn by the page's own `InputGroup::plain_sections`,
  `advanced_sections` and `InputSection::heading_in`; flags for changed from default, assumption,
  Advanced and Key design, whose group is not repeated, so every input is on one row; notes for a
  selector's choice, a blank optional input or empty text, and the free variable Torque -> Magnets
  sets); **Results** by physics chain as the table groups them (its own `table_lines`; each heading
  with the worst level of its checks; per result the value, unit, check level (`entry_level`),
  workbook cell, path, corrections and, where a record exists, the equation as plain text); **Assumptions** (value, unit, workbook default,
  rationale, source). Values only: no formula, and nothing reads the file back (X-1). Numbers are
  number cells at full precision, a number that is not finite is `+inf`, `-inf` or `NaN` text as in
  the CSV, and a text longer than a cell's 32,767 characters (as Excel counts them, in UTF-16 code
  units: an emoji counts two) is cut at a whole character, saying how long it was. The Inputs,
  Results and Assumptions sheets freeze their header row and carry Excel's filter buttons on it
  (the Summary, read top to bottom, does neither); the columns are sized, the headings bold (a
  group heading on a light grey, E7E6E6, apart from the section headings inside it) and each check
  level filled with the page's badge colour (green 5AC878, amber FF8F00, red FF0000). rust_xlsxwriter 0.99.1 writes
  it; on wasm32 its `wasm` feature is on, without which the workbook's creation time calls
  `SystemTime::now()` and panics in the browser. The host gets the file as bytes
  (`PanelRequest::SaveFile.contents` is a `Vec<u8>`).
````

In `magcoupling-rs/README.md`, replace (part of one line):

````markdown
`results_table.rs` (`table_entries`, `entry_index`, `search`, `ResultOrder`, `Line`, `table_lines`, `empty_text`, `results_csv`, `results_json`, `column_widths`), 
````

with:

````markdown
`results_table.rs` (`table_entries`, `entry_index`, `search`, `ResultOrder`, `Line`, `table_lines`, `empty_text`, `results_csv`, `results_json`, `column_widths`, `entry_level` and `worst_level` (a row's and a heading's badge), `RUST_ONLY`), `spreadsheet.rs` (the spreadsheet's layout: `sheets` of a `Snapshot` -> `Sheet` rows of `Cell`s with a `CellStyle`, `value_cell`, `level_cell`, the column headers), `xlsx.rs` (`spreadsheet_bytes`, `xlsx_bytes` with rust_xlsxwriter, `cell_text`, `MAX_CELL_CHARS`, `fill`, `GROUP_FILL`, `now_unix_s`, `XLSX_FILE_NAME`, `XLSX_MIME`), 
````

In `magcoupling-rs/README.md`, replace (part of one line):

````markdown
`inputs.rs` (`InputCatalogue` with `groups_in` and `section_of`, 
````

with:

````markdown
`inputs.rs` (`InputCatalogue` with `groups_in` and `section_of`, `InputGroup` with `plain_sections`, `advanced_sections` and `has_advanced`, `InputSection::heading_in` (how the inputs side, the filter's runs and the spreadsheet order and head a group's sections), 
````

In `magcoupling-rs/README.md`, replace (part of one line):

````markdown
every view draws beside the panel in a tiny window |
````

with:

````markdown
every view draws beside the panel in a tiny window; both spreadsheet buttons (the header's and the results table's) queue `magcoupling-results.xlsx`, which reads back with the four sheets, the edited face gap as a number and the share link, follows the input order shown and, in Torque -> Magnets, holds the solved length (noted) and the solve's outcome |
````

In `magcoupling-rs/README.md`, replace (part of one line):

````markdown
in a short region the results table scrolls in what its search and filter rows leave |
````

with:

````markdown
in a short region the results table scrolls in what its search and filter rows leave; a group's plain then advanced sections and their headings, as the inputs side, the filter and the spreadsheet draw them; a row's badge is its check's level (none for a check greyed by the end effect), a heading's the worst of its rows |
| `src/gui/spreadsheet.rs`, `src/gui/xlsx.rs` (feature `gui`) | The four sheets in order, the table sheets' column headers frozen with filter buttons (the Summary neither); every input on one row with its value, unit and cell in either order, in exactly the order the page draws them, under its group, section and Advanced headings; the flags (changed, assumption, Advanced, Key design) and the notes (a choice, a blank optional input or empty text, the sized variable); every result on one row and every heading as the table's grouped lines (`table_lines`) give them, each value at full precision with its corrections; each check's level, none for a check greyed by the end effect, and each heading's worst level; numbers that are not finite as text; the equation column exactly for the explained results; the Summary (export time, version, link, sizing, both banners, the dashboard's rows, the warnings, the corrections applied); the Assumptions sheet; the file read back with calamine cell for cell for three designs; frozen panes and filters where the sheet asks, every column's width, bold headings, the three fills and the group grey, and no formula in the XML; a text over 32,767 UTF-16 units (letters or emoji) cut at a whole character, saying its length; the fill colours; the clock |
````

In `magcoupling-rs/README.md`, replace (part of one line):

````markdown
the default link stays under 2,500 characters.
````

with:

````markdown
the default link stays under 2,500 characters, and the one 49fbd3f wrote (before the spreadsheet export's zlib-rs backend) still loads.
````

In `magcoupling-rs/README.md`, replace:

````markdown
- The window does the panel's requests: saving with the linkage app's download helper (a file
  dialog natively, a browser download on the web), "Load design" through rfd (the linkage web
````

with:

````markdown
- The window does the panel's requests: saving with the linkage app's download helper as bytes
  (`download_bytes`: a file dialog natively, with an Excel workbook filter for the spreadsheet, a
  browser download on the web), "Load design" through rfd (the linkage web
````

In `docs/ai/02-system.yaml`, replace (part of one line):

````yaml
The panel does no I/O: saving and picking files are PanelRequests its host does (the standalone app: app::files; the linkage app: gui::calculator_window, export::download to save and rfd to pick, natively and on the web)."
````

with:

````yaml
The panel does no I/O: saving and picking files are PanelRequests its host does (the standalone app: app::files; the linkage app: gui::calculator_window, export::download to save and rfd to pick, natively and on the web). A SaveFile carries the file's bytes: the design file and the CSV and JSON exports as UTF-8 text, the spreadsheet as an .xlsx file."
      - "The spreadsheet export (gui::spreadsheet lays it out, gui::xlsx writes it) holds values only, never a formula, and nothing reads it back into the site (decision X-1). Every input and every result is on exactly one row (the Key design inputs are flagged, not repeated), the inputs in the order the inputs side shows and the results in the results table's grouped order, both laid out by the page's own code (InputGroup::plain_sections and advanced_sections, InputSection::heading_in; results_table::table_lines, entry_level and worst_level), never a copy of its rules; a number that is not finite is the CSV's text; a check's level is check_level's, so a check greyed by the end effect has none."
````

In `docs/ai/02-system.yaml`, replace (part of one line):

````yaml
; next: M3 (live 3D fields)"
````

with:

````yaml
. Spreadsheet export built on branch magcoupling/xlsx (the header's Export spreadsheet and the results table's Export XLSX: an .xlsx of the page's layout, values only; decisions X-1 to X-12); next: M3 (live 3D fields)"
````

In `docs/ai/02-system.yaml`, replace:

````yaml

current_statuses:
````

with:

````yaml
  - "rust_xlsxwriter 0.99.1: Workbook::new() builds DocProperties::new(), which stamps the creation time with SystemTime::now() unless its wasm feature is on and the target is wasm32; without the feature the first workbook panics in the browser. magcoupling-rs declares the same optional dependency in [dependencies] and in its wasm32 target table with features = [\"wasm\"], activated by one dep:rust_xlsxwriter in the gui feature; cargo merges the features per target, so native builds go without it."
  - "Excel takes a hyperlink of at most 2,080 characters (rust_xlsxwriter refuses a longer one: MaxUrlLengthExceeded). A magcoupling share link names every input and is about 2,500 characters, so the spreadsheet writes it as text."
  - "zip 8's deflate feature (the zip of rust_xlsxwriter and calamine) is zopfli plus flate2's zlib-rs backend, and cargo's feature unification switches flate2 to zlib-rs in every build that carries the magcoupling panel, the linkage app's share links included. The compressed bytes are the backend's: at the default design the magcoupling payload differs from the one 49fbd3f wrote (both 2,462 characters), and so does a linkage link; tests pin a link each app wrote at 49fbd3f and check that it still decodes."

current_statuses:
````

In `docs/ai/03-structure.yaml`, replace (part of one line):

````yaml
src/gui/ (egui 0.32, egui_plot 0.33, serde_json, base64, flate2, log; no eframe, no I/O):
````

with:

````yaml
src/gui/ (egui 0.32, egui_plot 0.33, serde_json, base64, flate2, log, rust_xlsxwriter 0.99.1 (its wasm feature and js-sys on wasm32); no eframe, no I/O):
````

In `docs/ai/03-structure.yaml`, replace (part of one line):

````yaml
header with Undo/Redo/Reset all/Save design/Load design/Copy share link,
````

with:

````yaml
header with Undo/Redo/Reset all/Save design/Load design/Copy share link/Export spreadsheet,
````

In `docs/ai/03-structure.yaml`, replace (part of one line):

````yaml
PanelRequest (SaveFile, OpenDesign) drained by take_requests;
````

with:

````yaml
PanelRequest (SaveFile with the file's bytes, OpenDesign) drained by take_requests; export_spreadsheet (both spreadsheet buttons: the header's and the results table's Export XLSX); assumptions_banner;
````

In `docs/ai/03-structure.yaml`, replace (part of one line):

````yaml
row_tooltip, column_widths),
````

with:

````yaml
row_tooltip, column_widths, entry_level (a row's badge) and worst_level (a group heading's: the worst of its rows'), RUST_ONLY, TableAction (ExportCsv, ExportJson, ExportXlsx), EXPORT_XLSX),
````

In `docs/ai/03-structure.yaml`, replace (part of one line):

````yaml
history.rs (History, MAX_UNDO_LEVELS 100),
````

with:

````yaml
history.rs (History, MAX_UNDO_LEVELS 100), spreadsheet.rs (the spreadsheet export's layout, decisions X-1 to X-12: sheets(&Snapshot) -> Summary, Inputs (in the order shown), Results (by physics chain), Assumptions, each a Sheet (frozen_rows, autofilter) of rows of Cell {CellValue (Empty, Number, Text, Time), CellStyle (Plain, Title, Header, Group, Section, Level)}; value_cell, level_cell; the Inputs ordered and headed by the page's own InputGroup::plain_sections/advanced_sections and InputSection::heading_in, the Results by results_table::table_lines and entry_level; INPUT_COLUMNS, RESULT_COLUMNS, DASHBOARD_COLUMNS, ASSUMPTION_COLUMNS), xlsx.rs (spreadsheet_bytes, xlsx_bytes (rust_xlsxwriter: number cells, frozen headers and autofilters where the sheet asks, widths, bold headings, GROUP_FILL on the group headings, level fills), cell_text (cut at MAX_CELL_CHARS 32,767 UTF-16 code units, as Excel counts, at a whole character), fill (the badge colours), now_unix_s (js_sys::Date on wasm32), XLSX_FILE_NAME magcoupling-results.xlsx, XLSX_MIME),
````

In `docs/ai/03-structure.yaml`, replace (part of one line):

````yaml
assert_glyphs, short_magnets)"
````

with:

````yaml
assert_glyphs, short_magnets, read_xlsx (calamine, a dev-dependency), xlsx_part (zip))"
````

In `docs/ai/03-structure.yaml`, replace (part of one line):

````yaml
src/app/files.rs (save: rfd dialog natively, Blob download on the web;
````

with:

````yaml
src/app/files.rs (save of a request's bytes: rfd dialog natively (write_file), Blob download on the web;
````

In `docs/ai/03-structure.yaml`, replace (part of one line):

````yaml
11 lock parity (egui, egui_plot, eframe, wasm-bindgen; deploy-web.yml CLI pin)
````

with:

````yaml
11 lock parity (egui, egui_plot, eframe, rust_xlsxwriter, wasm-bindgen; deploy-web.yml CLI pin)
````

In `docs/ai/03-structure.yaml`, replace (part of one line):

````yaml
does the panel's requests (export::download, DesignPicker over rfd)
````

with:

````yaml
does the panel's requests (export::download's download_bytes, with an Excel workbook filter for the .xlsx spreadsheet; DesignPicker over rfd)
````

In `docs/ai/03-structure.yaml`, replace (part of one line):

````yaml
inputs.rs (InputCatalogue::get with groups (the workbook order) and workflow, groups_in(InputOrder), section_of(InputOrder, path);
````

with:

````yaml
inputs.rs (InputCatalogue::get with groups (the workbook order) and workflow, groups_in(InputOrder), section_of(InputOrder, path); InputGroup::plain_sections, advanced_sections and has_advanced, InputSection::heading_in (how the inputs side, the filter's runs and the spreadsheet order and head a group's sections);
````

In `docs/ai/04-memory.yaml`, replace:

````yaml
  - "NEXT (user request 2026-10-03): a download button for the spreadsheet equivalent of the current website layout (an .xlsx that mirrors the displayed input groups and result groups, ideally with the formulas)"
````

with:

````yaml
  - "Magcoupling spreadsheet export (user request 2026-10-03, clarified: values only, no live formulas, no round trip from Excel back to the site; plan docs/superpowers/plans/2026-10-03-magcoupling-xlsx-export.md, branch magcoupling/xlsx): built with the recommended option of each decision X-1 to X-12 (Summary, Inputs in the order shown with the Key design flagged, Results by physics chain with the equations as plain text, Assumptions; the share link as text; magcoupling-results.xlsx; the header's Export spreadsheet and the results table's Export XLSX; rust_xlsxwriter 0.99.1, its wasm feature on wasm32; SaveFile carries bytes). Before merge the user confirms the plan's Decisions table and opens a downloaded file in Excel (or LibreOffice)."
````

In `docs/ai/05-update-tracker.md`, replace:

````markdown
---

## 2026-10-03 — Magcoupling GUI: views stay in their own area (branch magcoupling/overlap-fix)
````

with:

````markdown
---

## 2026-10-03 — Magcoupling GUI: spreadsheet download of the current layout (branch magcoupling/xlsx)
- "Export spreadsheet" (header) and "Export XLSX" (beside the results table's CSV and JSON exports) save `magcoupling-results.xlsx`: the design shown laid out as the page, values only (decision X-1: no formulas, no round trip). Sheets Summary (export time, version, the share link as text, the sizing state, the banners, the dashboard with badges, the material warnings, the corrections applied), Inputs (the order the inputs side shows; group, section and Advanced headings; changed, assumption, Advanced and Key design flags; every input once), Results (by physics chain with heading levels, check levels as the page's fills, the equation as plain text where a record exists; every result once) and Assumptions. `gui::spreadsheet` lays the sheets out (tested directly), `gui::xlsx` writes them with rust_xlsxwriter 0.99.1 (pinned; gate 11 checks both lock files), read back in tests with calamine and zip (dev-dependencies). The sheets' order and headings come from the page's own code (`InputGroup::plain_sections`/`advanced_sections`, `InputSection::heading_in`, `results_table::table_lines`, `entry_level`), which the inputs side and the results table call too; the table sheets freeze and filter their header row, group headings are grey, and a text past Excel's 32,767 UTF-16 units is cut at a whole character. Share links (both apps) are now compressed by flate2's zlib-rs backend (rust_xlsxwriter's zip); tests pin a link each app wrote before and check it still decodes. `PanelRequest::SaveFile.contents` is now `Vec<u8>`; both hosts write the bytes (app::files::write_file; linkage export::download::write_file through download_bytes, with an Excel workbook save filter). The plan records the decisions table and the measured web bundle growth.

## 2026-10-03 — Magcoupling GUI: views stay in their own area (branch magcoupling/overlap-fix)
````


Then copy this plan into the worktree:

```bash
mkdir -p C:/Users/Cole/source/repos/lsim-mag-xlsx/docs/superpowers/plans
cp C:/Users/Cole/AppData/Local/Temp/claude/C--Users-Cole-source-repos-linkage-simulation/3165685f-bf49-42bb-a036-671c0e065429/scratchpad/xlsx-plan/plan.md C:/Users/Cole/source/repos/lsim-mag-xlsx/docs/superpowers/plans/2026-10-03-magcoupling-xlsx-export.md
```

- [ ] **Step 6: Check the docs, format, lint and gate**

Run:

```bash
python -c "import yaml; [yaml.safe_load(open(f, encoding='utf-8')) for f in ['C:/Users/Cole/source/repos/lsim-mag-xlsx/docs/ai/03-structure.yaml', 'C:/Users/Cole/source/repos/lsim-mag-xlsx/docs/ai/04-memory.yaml']]; print('yaml ok')"
git -C C:/Users/Cole/source/repos/lsim-mag-xlsx diff --check
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-xlsx/magcoupling-rs/Cargo.toml --check
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-xlsx/linkage-sim-rs/scripts/gate.sh 2>&1 | tail -n 3
git -C C:/Users/Cole/source/repos/lsim-mag-xlsx checkout -- docs/chebyshev_lambda
```

Expected: `yaml ok` (`02-system.yaml` is left out: it does not parse at `49fbd3f`, before this plan, and this plan's items in it each parse on their own); `diff --check` prints nothing; `fmt --check` prints nothing; the gate ends `GATE PASS` with no `SKIP gate` line.

- [ ] **Step 7: Build both web bundles and download the file in a browser**

Run:

```bash
bash C:/Users/Cole/source/repos/lsim-mag-xlsx/linkage-sim-rs/scripts/build_web.sh 2>&1 | grep -E "parity guard|complete|bg.wasm"
ls -l C:/Users/Cole/source/repos/lsim-mag-xlsx/linkage-sim-rs/web/linkage-web_bg.wasm C:/Users/Cole/source/repos/lsim-mag-xlsx/linkage-sim-rs/web/magcoupling/magcoupling-web_bg.wasm
```

Expected: both guards print `builds without workbook-parity`, both builds `complete!`; the sizes about `linkage-web_bg.wasm` 13,189,224 bytes and `magcoupling-web_bg.wasm` 6,219,376 bytes (12,115,195 and 5,118,840 at `49fbd3f`) (Verification record).

Then serve `C:/Users/Cole/source/repos/lsim-mag-xlsx/linkage-sim-rs/web` with `python -m http.server 8093` in the background (its Windows PID, which Git Bash's `$!` is not: PowerShell `(Get-NetTCPConnection -LocalPort 8093 -State Listen).OwningProcess`), open `http://localhost:8093/magcoupling/` with Playwright at 1400 x 900, wait for the console line `magcoupling explorer: ` (the registry is built), take a screenshot, click the header's "Export spreadsheet" at its position on the canvas (egui draws on a canvas: click by coordinates, `page.mouse.click(x, y)`), and save the download (`page.waitForEvent('download')`, `download.saveAs(...)`; in Python Playwright `page.expect_download()` and `save_as`, with `p.chromium.launch(channel="chrome")` if its own browser is not installed; in this plan's replay the Playwright MCP's code tool lost its page on the download, Python Playwright did not) as `C:/Users/Cole/AppData/Local/Temp/claude/C--Users-Cole-source-repos-linkage-simulation/3165685f-bf49-42bb-a036-671c0e065429/scratchpad/xlsx-plan/sample.xlsx`. Open it with the system Python:

```bash
python -c "import openpyxl; wb = openpyxl.load_workbook(r'C:/Users/Cole/AppData/Local/Temp/claude/C--Users-Cole-source-repos-linkage-simulation/3165685f-bf49-42bb-a036-671c0e065429/scratchpad/xlsx-plan/sample.xlsx'); print(wb.sheetnames); s = wb['Summary']; print(s['A1'].value, '|', s['A2'].value, s['B2'].value, '|', s['A4'].value, str(s['B4'].value)[:40]); i = wb['Inputs']; print([c.value for c in i[1]][:5]); print(s.freeze_panes, wb['Results'].freeze_panes, i.freeze_panes, wb['Assumptions'].freeze_panes); print(i.auto_filter.ref, wb['Results'].auto_filter.ref, wb['Assumptions'].auto_filter.ref, s.auto_filter.ref); print(i['A2'].value, i['A2'].fill.fgColor.rgb, i['A2'].font.b)"
```

Expected: the download is `magcoupling-results.xlsx`, the header's status line says `Downloaded magcoupling-results.xlsx`, the browser console has no error or warning, and Python prints (the export time and the link's start differ): the four sheets; the title, the export time and the link; the Inputs header; the Summary not frozen and the other three frozen at A2; the three table sheets' filters over every row and column and none on the Summary; the first group heading bold on the light grey:

```
['Summary', 'Inputs', 'Results', 'Assumptions']
Magnetic coupling calculator | Exported (UTC) 2026-10-03 21:17:21 | Share link (paste into a browser) http://localhost:8093/magcoupling/?m=lVh
['Input', 'Value', 'Unit', 'Workbook cell', 'Path']
None A2 A2 A2
A1:J215 A1:H1107 A1:H16 None
Requirements and operating conditions FFE7E6E6 True
```
 Stop the server: PowerShell `Stop-Process -Id <that PID> -Force`, and check that `Get-NetTCPConnection -LocalPort 8093 -State Listen` finds nothing.

- [ ] **Step 8: Commit**

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-xlsx add magcoupling-rs/src/gui/panel.rs magcoupling-rs/src/gui/results_table.rs linkage-sim-rs/src/gui/calculator_window.rs magcoupling-rs/README.md docs/ai/02-system.yaml docs/ai/03-structure.yaml docs/ai/04-memory.yaml docs/ai/05-update-tracker.md docs/superpowers/plans/2026-10-03-magcoupling-xlsx-export.md
git -C C:/Users/Cole/source/repos/lsim-mag-xlsx commit -m "feat(magcoupling-rs): spreadsheet buttons in the header and the results table, the linkage save filter, docs (decision X-8)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9"
git -C C:/Users/Cole/source/repos/lsim-mag-xlsx status --short
```

Expected: one commit; no status lines (`linkage-sim-rs/web/*.wasm` and `*.js` are gitignored).

After Task 4: the whole-branch review on the session model (`// session model: final whole-branch review`), then superpowers:finishing-a-development-branch. Before merge the user confirms the Decisions table and opens a downloaded file in Excel.

## Self-review record

The critic's review of the first draft found 2 blocking defects (both DRY: the spreadsheet re-derived the page's own ordering code) and 7 non-blocking points. Each is fixed in this revision; the replay above verifies the revised blocks. The user's request ("No need for live formulas from excel back to site") is unchanged: X-1 stays values only.

| # | Finding | What changed | Evidence |
|---|---|---|---|
| B1 | `results_sheet` walked `result_groups()` itself, with its own `other_started` flag: a copy of `table_lines`' grouped walk (and unlike it, it would head an empty group) | `results_sheet` iterates `table_lines(&all, ResultOrder::Grouped, None, &\|_\| true)`: `Line::OtherResults` is the "Other results" heading, `Line::Group` a group's heading with `worst_level`, `Line::Row` a result's row (`result_row`). `other_started` is gone; an empty group is now left out exactly as the table leaves it out | Task 2 blocks; `every_result_is_on_one_row_by_physics_chain_with_its_value` now takes both the rows and the headings (with their styles) from `table_lines`' output, plus the headline first and "Other results" once |
| B1b | The row's check level was copied from `ResultsTable::ui` (`entry.check.then(\|\| check_level(..)).flatten()`), and `worst_level` repeated its filter | `results_table::entry_level(&DesignResults, &TableEntry) -> Option<Level>`; `worst_level` is `filter_map(entry_level).max()`; the table's `Line::Row` arm and the sheet's `result_row` both call `entry_level` | Task 2 blocks (the table's row arm is its own block); the new `a_row_carries_its_check_s_level_and_a_heading_the_worst_of_its_rows`; mutations "every row red" and "every level dropped" fail |
| B2 | `inputs_sheet` copied the page's split (plain sections, Advanced heading, advanced sections) and the rule that a group's own section has no heading, which then lived in three places | `InputGroup::plain_sections`, `advanced_sections`, `has_advanced` and `InputSection::heading_in`, used by the panel's `inputs_ui` and `section_ui`, by `SectionMatches::heading` and by `inputs_sheet`/`section_rows` | Task 2 blocks in `inputs.rs` and `panel.rs`; the new `a_group_lists_its_sections_and_their_headings_as_the_page_draws_them`; the mutation "own section headed" fails it and the sheet's test |
| T | No test pinned the input order within a section (`.rev()` passed all 19 tests) | `the_inputs_follow_...` asserts the exact sequence of path cells, in both orders, against `plain_sections().chain(advanced_sections())`; it also counts the section headings (one per section but the group's own, plus the Advanced ones) and checks that none repeats its group's | the critic's `.rev()` mutation now fails it; so does "own section headed" (it survived the first strengthening, which is why the heading count was added) |
| I | Global Constraints said share links "keep their bytes", contradicting X-9 | Reworded: design files and the CSV and JSON exports keep their bytes; share links are compressed by zlib-rs after Task 3 and still decode. Measured: the default design's payload differs from `49fbd3f`'s from its fourth character (both 2,462), a linkage link from its 27th. Task 3 adds `a_share_link_written_before_the_zlib_rs_backend_still_loads` (the `49fbd3f` payload decodes to `Design::default()`) and, for the linkage app's own links, `share_url_written_before_the_zlib_rs_backend_still_decodes`; the 02-system lesson that said no test pins a link is corrected | Task 3 Steps 4 and 5 run both; the payloads were generated by throwaway tests on a `49fbd3f` export |
| R | Only rust_xlsxwriter is pinned; Task 3 expected exact `Adding` lines | The informational option: the Global Constraints and Task 3 Steps 2 and 5 say the `Adding` lines and the zip version are for information; only rust_xlsxwriter not at `0.99.1` or wasm-bindgen not at `0.2.114` escalates. Pinning was rejected because an exact dev-dependency pin in magcoupling-rs cannot reach `linkage-sim-rs/Cargo.lock` (cargo does not resolve a dependency's dev-dependencies) | cargo's own line, `Locking 13 packages to latest Rust 1.89.0 compatible versions`, shows the resolver also keeps to the toolchain |
| U | Group and section headings hard to tell apart; the Summary froze only its title; no filters on 1,107 rows | A group heading is bold 12 pt on `GROUP_FILL` (E7E6E6); `Sheet` gains `autofilter`; the Summary has `frozen_rows: 0` and no filter, the Inputs, Results and Assumptions sheets freeze row 1 and filter it over every row and column. Folded into decision X-11 (now "The check colours and the headings") rather than a new id, so the docs' "X-1 to X-12" stay true | `the_file_has_frozen_headers_filters_widths_bold_headings_fills_and_no_formula` reads the panes, the `<autoFilter>` refs and the grey fill from the XML; the sample shows them (Verification record) |
| Ea | An empty text input (a part name left out) wrote an empty Value cell with no note | It is noted `blank` (`BLANK_TEXT`, the word the page uses for a blank optional input). The page's `text_hint` was considered and not used: it would add notes to every text input, beyond the finding | `an_input_row_flags_...` checks the note, and that a part name with a value has none; the mutation "empty text not noted" fails it |
| Eb | `cell_text` counted chars, Excel counts UTF-16 units | `cell_text` measures and cuts in UTF-16 units, after the last whole character that leaves room for the note (which still says how many characters the text had) | 20,000 emoji (40,000 units) are cut to 32,766 or 32,767 units of whole emoji and read back through calamine; a text at the limit is kept, one unit over is cut; the mutation "counts chars" fails |
| — | Sizes and model tiers: no change needed | X-9 and Task 4 Step 7 carry this replay's sizes (the bundles grew by 1,100,536 and 1,074,029 bytes); the model tiers are as before | Verification record |

Also found while revising, and fixed or recorded:

- The critic's own mutation run had restored `spreadsheet.rs` in the verify tree with CRLF endings (that repository had `core.autocrlf=true`); the scratch repositories now set `core.autocrlf=false`, and every replayed file was compared byte for byte.
- `linkage-sim-rs` is not rustfmt-clean, so the new linkage test was checked alone with `rustfmt --edition 2024`: its first layout split the `assert_eq!` differently from rustfmt, and the block now has rustfmt's.
- Task 0 Step 2's list of files that must still be `49fbd3f`'s now includes `linkage-sim-rs/src/gui/state/file_io.rs`, which Task 3 now touches; Task 0 Step 4 reads the new helpers' neighbours (`table_lines`, `SectionMatches::heading`, the share payload functions).
- The window movement the first replay could not reproduce is pinned down (Verification record): only a scripted instant click on "Export spreadsheet" moves the window; a person's click does not. Left for the user's hand check; no code here touches the window.
- The plan has 77 blocks (64 before): the new tests and helpers in `inputs.rs` and `results_table.rs`, the panel's use of the helpers, the two pinned-link tests and the extra docs replacements.
