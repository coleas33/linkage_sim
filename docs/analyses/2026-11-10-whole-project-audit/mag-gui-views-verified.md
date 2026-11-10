# Verified: mag-gui-views

## MGV-01: The results table and the clamp tab draw the catalogue label; every other view draws the display label
- Verdict: CONFIRMED (the code reads as described; whether it is a defect is the owner's call)
- Checked: results_table.rs:770-771 draws `entry.info.meta.label` in the row; clamp_drawing.rs:776 (screw_rows keeps `entry.info.meta.label`), 827 and 862 draw the catalogue label. The table's search haystack uses `display_label(&row.path, info.meta.label)` plus the catalogue label (results_table.rs:142-147) and its test (946-965) pins only the search, not the drawn label. dashboard.rs:530 hover uses `display_label`, dashboard.rs:748 draws `label_for`. plain_text.rs:851-860 defines both; LABELS renames results (four of them differ from the catalogue per the test). Grep for label_for/display_label over magcoupling-rs/src (about 170 hits) shows no other drawn result label using the catalogue label.
- Safe to apply without judgment: no, it changes what the user sees, and spec 3.5 and the planned UX overhaul decide which label the table shows.

## MGV-02: Stored-3D path lists typed twice, and the dashboard copies are used only by tests
- Verdict: CONFIRMED (both parts: used only by tests; the 14 paths are typed in both places)
- Checked: Grep STORED_3D_INPUTS|STORED_3D_SENTENCE over the whole worktree. Non-doc hits in magcoupling-rs/src: dashboard.rs:18 (doc link), 235 (doc), 237, 258 (definitions), tests at 1642/1657/1668/1669/1685/1725/1728 (all after `#[cfg(test)]` at 1034), plain_text.rs:1531 (after `#[cfg(test)]` at 1423), engine/temperature.rs:55 (doc comment naming `gui::dashboard::STORED_3D_SENTENCE`). No hit in tests/, scripts, web JS or HTML. README.md:351 and docs/ai yaml/md mention the names; the rest are historical plans and specs. inputs.rs:576-598 lists the same 14 paths in `calibration.stored_demag` (4) and `calibration.stored_slip` (10).
- Safe to apply without judgment: no. Moving or deleting the constants breaks the rustdoc links at dashboard.rs:18, 235 and engine/temperature.rs:55, the README line 351 and docs/ai text name them, and plain_text.rs:1531 (a different file's test) reads the dashboard constant, so the choice (test-module move vs build from the two sections) is a judgment.

## MGV-03: Two `pub` functions in shipped code are called only from tests
- Verdict: CONFIRMED
- Checked: Grep `as_text|package_label` over the whole worktree (6 files). `Cell::as_text` is defined at spreadsheet.rs:196; every other hit in that file is at line 599 or later, after `#[cfg(test)] mod tests` at 572 (`use super::*` at 574). `package_label` is defined at result_groups.rs:136; its only other use is line 389 inside `mod tests` (226, `use super::*` at 228); `groups_of` iterates `PACKAGE_LABELS` directly (204). No hit in the other crate, tests/, scripts, web or yaml; the other four files are historical plans. Results_table.rs:1107 is only a test name containing "as_text".
- Safe to apply without judgment: yes. Nothing outside the two test modules calls either, a private test-only copy in the test module behaves the same, and no target or feature differs. (Only the shipped library's pub surface shrinks; nothing in the worktree uses it.)
- Exact change:
  - magcoupling-rs/src/gui/spreadsheet.rs lines 194-201: delete the blank line 194 and this block
    ```
        /// The cell's text; empty for a cell that holds no text.
        pub fn as_text(&self) -> &str {
            match &self.value {
                CellValue::Text(text) => text,
                _ => "",
            }
        }
    ```
    (the `impl Cell` then ends after `styled`). Then in `mod tests`, after line 579 (the last `use`), add
    ```

        impl Cell {
            fn as_text(&self) -> &str {
                match &self.value {
                    CellValue::Text(text) => text,
                    _ => "",
                }
            }
        }
    ```
  - magcoupling-rs/src/gui/result_groups.rs lines 135-142: delete
    ```
    /// The heading of a package; `None` if [`PACKAGE_LABELS`] lacks it (a test checks none does).
    pub fn package_label(package: &str) -> Option<&'static str> {
        PACKAGE_LABELS
            .iter()
            .find(|(id, _)| *id == package)
            .map(|(_, label)| *label)
    }

    ```
    and in `mod tests` after `use super::*;` (line 228) add
    ```

        /// The heading of a package; `None` if [`PACKAGE_LABELS`] lacks it.
        fn package_label(package: &str) -> Option<&'static str> {
            PACKAGE_LABELS
                .iter()
                .find(|(id, _)| *id == package)
                .map(|(_, label)| *label)
        }
    ```

## MGV-04: The Equation panel reads `clamps::PAUSED` while the panel passes `clamps_paused` everywhere else
- Verdict: CONFIRMED
- Checked: explorer.rs:132-133 (`START_HERE_HOVER = start_here_hover(clamps::PAUSED)`), 745 (`used_by_shown(&path, clamps::PAUSED)`), 2639 and 2701 (tests); `used_by_shown` and `start_here_hover` already take a `clamps_paused` argument (explorer.rs:123, 186). panel.rs:415-417 documents the field ("a field, not the constant"), 464 sets it from `clamps::PAUSED`, 1938-1942 (harness comment) says the explorer reads the constant itself. About 30 panel.rs sites pass `self.clamps_paused`. In the shipped app both equal the constant, so behaviour is identical today.
- Safe to apply without judgment: no. `explorer_ui` and its callers and tests need a new parameter, and the const `START_HERE_HOVER` has users; the signature choice is a judgment.

## MGV-05: plots.rs matches the sweep status by literal strings the engine owns
- Verdict: CONFIRMED
- Checked: plots.rs:322-326 matches `"nominal: test needed"` and `"below hot minimum"` literally, `_` goes to `no_fit`; engine/sweeps.rs:172-176 returns the same two literals (and three more: "inner flat too narrow", "outer flat too narrow", "outside OD envelope"). Grep of both strings over the worktree: other users are tests/differential.rs:276-288 and tests/deviations.rs:535, 564 (value pins, which should keep literals), the Python oracle, and docs. No constants exist for them (no `STATUS_` in sweeps.rs; plots.rs imports only `SweepRow` from sweeps at line 48).
- Safe to apply without judgment: yes. The strings stay byte-equal, so behaviour is unchanged on every target and feature, and only these two places change. A mistake (a missing import) would turn the arms into catch-all bindings, which rustc flags as an unreachable-pattern warning, and the plots tests (745-770) would fail.
- Exact change:
  - magcoupling-rs/src/engine/sweeps.rs after line 46 (`pub const POLE_SWEEP_POLES ...`): add
    ```

    /// The status of a row that fits and passes the hot minimum (read by `gui::plots`).
    pub const STATUS_NOMINAL: &str = "nominal: test needed";
    /// The status of a row that fits but falls below the hot minimum (read by `gui::plots`).
    pub const STATUS_BELOW_MINIMUM: &str = "below hot minimum";
    ```
  - sweeps.rs line 173: `"below hot minimum"` -> `STATUS_BELOW_MINIMUM`; line 175: `"nominal: test needed"` -> `STATUS_NOMINAL`.
  - plots.rs line 48: `use crate::engine::sweeps::SweepRow;` -> `use crate::engine::sweeps::{STATUS_BELOW_MINIMUM, STATUS_NOMINAL, SweepRow};`
  - plots.rs line 323: `"nominal: test needed" => points.nominal.push(p),` -> `STATUS_NOMINAL => points.nominal.push(p),`; line 324: `"below hot minimum" => points.below_minimum.push(p),` -> `STATUS_BELOW_MINIMUM => points.below_minimum.push(p),`

## MGV-06: Every dashboard guide recomputes its level and notes each frame, also when its block is shut
- Verdict: PARTLY. Holds: `indexed_guide_ui` (dashboard.rs:856-872) builds `guide_notes(...)` every frame (865) and the notes are read only in `fix_line` (934-936), drawn only when `open` (898-900); `guide_notes` allocates Vec<String> from trade_offs, side_effects and unchecked_costs (guidance.rs:1034-1046). Does not hold: the level is computed twice only for the Magnets fit row (fit_ui:829, then indexed_guide_ui:867). The other call site (dashboard.rs:770) goes straight to `indexed_guide_ui`, so the other guides compute the level once, not twice.
- Checked: dashboard.rs:770, 817-901, guidance.rs:668, 1023-1046; Grep for the five function names over dashboard.rs and guidance.rs. `guide_ui` decides `open` for amber only through `caution_header_ui` (888), so building the notes lazily means restructuring `GuideShown`, which tests construct directly (dashboard.rs:2686, 2904-2915).
- Safe to apply without judgment: no. Passing the level only helps one row, and lazy notes change `GuideShown`/`guide_ui`, used by tests.

## MGV-07: The same two axis lines are written out in three diagram functions
- Verdict: PARTLY. Holds for two functions: torque_angle (diagrams.rs:294-301) and heating_curve (502-509) open with the identical pair (bottom axis, then left axis, both `Stroke::new(1.0, pens.weak)`). The third (424-432) has the same bottom line but its second line is `[plot.right_bottom(), plot.right_top()]` (the right edge), so only one of the two lines is shared there.
- Checked: diagrams.rs:285-309, 420-436, 496-514; Grep of `plot.left_bottom()` (7 hits).
- Safe to apply without judgment: no. A helper changes how the shapes vec is built in three functions where shape order and count may be pinned by tests, and the third function does not fit the "two axes" helper.

## MGV-08: The `app` feature lists dependencies its `gui` feature already enables
- Verdict: CONFIRMED
- Checked: magcoupling-rs/Cargo.toml: `gui` (21-30) contains "dep:log" (24) and "dep:js-sys" (29); `app` (34-44) starts with "gui" (35) and repeats "dep:log" (37) and "dep:js-sys" (43). Both deps are optional deps enabled by `gui`, and enabling `app` enables `gui`, so the set of enabled dependencies is identical. Grep of `features =` and feature names across the worktree Cargo.toml files, scripts and yaml: `app` is only enabled via `required-features` (Cargo.toml:129, 134) and by name nowhere else with a conflicting meaning.
- Safe to apply without judgment: yes. Feature unification makes the entries no-ops, and Cargo.lock does not change.
- Exact change: magcoupling-rs/Cargo.toml, delete line 37 `    "dep:log",` and line 43 `    "dep:js-sys",` (both inside `app = [ ... ]`).

## Seen while checking
- (none)
