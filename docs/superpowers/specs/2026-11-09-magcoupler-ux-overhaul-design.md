# Magcoupler overhaul: an instrument that teaches — design

- Date: 2026-11-09. Base: `main` at 1048962. Status: **approved by the owner on 2026-11-16 ("Approve and yes. Stop asking for my direction and do what you think is best"); slice A (BL-077) is merged (17e4638); slice B is built in stages, and its first, the look (section 6; BL-079), is built by the plan `docs/superpowers/plans/2026-11-18-magcoupler-overhaul-b1-the-look.md`; the second stage, sections 4 and 5, is built in two parts, and its first (the check list, the Explain panel, the tiles; BL-081) by the plan `docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-the-instrument.md`; the second part is built in two, and its first (the top bar and the notes; BL-085) by the plan `docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-part2a-top-bar-and-notes.md`; part 2b is built in three, and its first (the Size column's lines in the Explain panel; BL-090) by the plan `docs/superpowers/plans/2026-11-23-magcoupler-overhaul-b2-part2b-i-size-column.md`; part 2b-ii is built in three, and its first (every input a selection; BL-092) by the plan `docs/superpowers/plans/2026-11-23-magcoupler-overhaul-b2-part2b-ii-a-inputs.md`; parts 2b-ii-b, 2b-ii-c and 2b-iii and slices C to F are not built.**
- Revised 2026-11-09 for the owner's ruling on the requirement (O-8, section 7.1) and for the builder check
  (section 12.5), and on 2026-11-10 for the owner's answer on the number (O-9) and the ruling on open questions (O-10).
- This is an umbrella design for six slices (section 3). It is firm on slices A and B and lighter on C to F. Each
  slice gets its own plan and branch (`magcoupling/overhaul-<letter>-<name>`); slices C to F each get a short design
  addendum before their plan. After approval the next step is the plan for slice A, and nothing else.
- Backlog: BL-077 is slice A's item (filed 2026-11-17 with its plan, `docs/superpowers/plans/2026-11-17-magcoupler-overhaul-a-first-arrival.md`). BL-079 is the item of slice B's first stage, the look (filed 2026-11-18 with its plan). Each later plan files one item with the next free id (`dimension: gui`; `risk: physics`
  wherever a slice shows a new number or a new physical claim, `risk: mechanical` otherwise).
- `M` = `magcoupling-rs`; code paths are under `M/src/` unless they say otherwise, and every `path:line` is at 1048962.
- Inputs: the inventory `docs/analyses/2026-11-09-magcoupler-ux-inventory.md` (what is on screen today, with its
  section "Constraints found 2026-11-09"); the mockups the owner chose from (gitignored, on this machine):
  `.superpowers/brainstorm/522-1794188235/content/first-arrival.html` (opening view, option B),
  `look-and-feel.html` (look, option 2) and `teaching-reveal.html` (the Explain panel); the last three specs under
  `docs/superpowers/specs/` (2026-10-13, 2026-10-14 and 2026-10-19), cited as spec1, spec2 and spec3.
- Numbers marked **(I)** come from the inventory's headless harness (the real panel at 1280x800). **(E)** come from
  `compute_all` at 1048962. **(H)** are hand calculations from the formula quoted beside them. **(M)** are the
  mockup's and bind nothing.
- "The owner" is the person who approves this document. "Mine" marks a call this design takes that the owner did not
  make; the owner may overrule any of them.

## 1. Purpose

### 1.1 What the owner asked for

> "I want to stop adding features for now and work on cleaning up the website, making it usable, less clunk and more
> straightforward to use. I want all the features avaiable to a user but teach them what each thing means in an
> effective way that is clean and professional."

A user should "see the diagram and intuitively understand what the purpose is for the tool". The audience is
mechanically literate engineers who are not magnetics specialists (spec1).

Later the owner asked how to make the tool "really useful and a good teaching tool into how the magcoupler works",
with "general improvements to math/function", and approved the whole list that came back ("Let's do it. All of these
sound good"). Section 1.2 lists it.

### 1.2 Owner decisions (2026-11-08 to 2026-11-10)

| Id | Decision |
|---|---|
| O-1 | Scope: the magnetic coupling calculator only. The linkage tool is not touched. |
| O-2 | Opening view: option B, "The instrument". The drawing and four numbers in the centre, the size controls on the left, the checks on the right. |
| O-3 | Teaching: one Explain panel is what a click opens, everywhere. Its first state is a short orientation. Longer lessons sit behind lesson links. |
| O-4 | Look: option 2, "Charcoal and crimson" (the tokens in section 6.1). |
| O-5 | Every feature that exists today stays available. |
| O-6 | The approved improvement list, items 1 to 11 below. |
| O-7 | Order: "the passing example, the address bar and the seven fixes first". |
| O-8 | The requirement of the opening design. The machine is "wheel, wheel bearings, mag coupler, 5:1 gearbox, belt, 1:1 gearbox, motor". "Motor drives wheel. Wheel bearing are around 0.5 Nm cold, or 0.35 Nm hot. Acceleration torque is as much as we can fit in this space, update the requirements to match what can be achieved with the tested/given default parameters". Section 7.1 says how this design reads it (C-19). |
| O-9 | The number (2026-11-10). Asked "Is 1.35 N·m the requirement you want the tool to open on?", the owner answered "Yes 1.35Nm". The starting design opens with Required torque at 1.35 N·m (section 7.1). |
| O-10 | Open questions (2026-11-10): "Any open questions choose the conservative answer for now and make those assumptions known somewhwre". Q-1, Q-3, Q-4 and Q-5 take their recommendations for now, as assumptions the owner can overrule; `docs/analyses/2026-11-10-conservative-assumptions.md` lists them (rows A-2 to A-5) with the other assumptions made under this ruling. This document reads the ruling as an answer to open questions, not as its own approval: the status line above stands, and no plan is written until the owner approves the document. |

The approved list (O-6):

1. Open on a curated design that passes its checks. `DesignInputs::default` stays as it is, for workbook parity.
2. Twist: drag the inner ring in the drawing, with a marker on the Torque vs rotation plot.
3. Replace the 14-step Start-here tour with a five-step worked example.
4. Lessons as web pages generated from the notes, with links that open the calculator at a design. The short Explain
   panel stays in the tool.
5. The design is written into the address bar on each edit.
6. A fix button on every failing check.
7. Gearbox, wheel speed and shaft joints as optional application sections.
8. Stored 3D fields that follow the geometry (its own project).
9. The seven fixes: BL-058, BL-061, BL-062, BL-063, BL-067, BL-068 and BL-074.
10. Uncertainty on the headline torque.
11. Later: torsional stiffness, start-up slip, bearing forces, sleeve hoop stress.

Items 8, 9 and 11 are not part of this design. Item 9 runs beside it as its own branch
(`magcoupling/off-default-fixes`); section 9.3 says how the two meet.

### 1.3 Calls this design takes (mine)

Each is a default the owner has not ruled on. The ones that change what a user sees are repeated in the hand-over
list.

| Id | Call | Why |
|---|---|---|
| C-1 | Slice A is the starting design and the address bar on **today's** screen. The new look and layout are slice B. | O-7 puts those two first; they do not depend on the redesign and are low risk. |
| C-2 | A click selects; a hover never changes the Explain panel. | A panel that follows the pointer flickers as the mouse crosses the screen. |
| C-3 | The check list keeps one fixed order (today's `CHECK_GUIDES` order). Failing checks are not sorted to the top. | Rows that jump while a slider is dragged are the clunk this work removes. |
| C-4 | A check's state is a shape and a word as well as a colour: a filled disc for fails, a triangle for caution, a ring for passes. | Colour alone fails colour-blind users, and crimson (brand) sits near red (fails). |
| C-5 | Crimson is never a text colour and never appears on a value or in the check list. It marks the brand mark, the active tab and the one primary button. Red means a failing check and nothing else. | Crimson text on the panel is 2.9:1 (H, section 6.1); and brand red must not read as an error. |
| C-6 | Numbers lead their labels on the four tiles; digits are tabular everywhere; borders are hairlines; no gradients, glows or idle animation. | The "instrument" of O-2. |
| C-7 | The drawing keeps its pole colours (warm for north, cool for south). The rings are named by position ("inner ring", "outer ring") and shown by an outline on hover. | The mockup's "drive ring red, driven ring blue" contradicts the real drawing, where red and blue are poles. |
| C-8 | Nothing is stored between visits. The address bar is the only memory. | No consent question, no stale state, one mechanism. |
| C-9 | The five plots sit under one "Plots" tab with a picker. The tab row becomes Size and torque, Geometry, Plots, Results. | Eight tabs wrap at 1280 px; five of them are plots. |
| C-10 | The torque card moves into the Explain panel: the whole card under the Hot-low tile, one row under each size control. `TorqueCard::lines` is unchanged. | The card is 163 of the 727 first-paint words (I). |
| C-11 | The Equation panel stays where it is (docked at the bottom) and opens from the Explain panel's "Equation" link. Its "Explain" checkbox goes: the note for the open equation shows in the Explain panel. | One place for words; no second thing called Explain. |
| C-12 | "Show spreadsheet details" moves into the Design menu. | It is a rare switch, not a top-bar control. |
| C-13 | Links keep egui's link blue, the hue of the drawing's dimension lines. Blue means "click me" across the tool. | The approved tokens name no link colour; the dimension lines are already clickable and blue. |
| C-14 | The muted text token is lightened from `#7c838c` to `#8a919a`. | `#7c838c` on the raised panel colour is 4.2:1, under the 4.5:1 floor; `#8a919a` is 5.0:1 (H, section 6.1). |
| C-15 | Slip heating is always judged. Only the Gearbox and Shaft joints sections can be switched off (slice D). | Slip heat decides whether the magnets survive, which is the coupling's own limit, not the application's. **This departs from item 7's wording ("wheel speed"); see Q-3.** |
| C-16 | "Uncertainty on the headline torque" (item 10) is drawn as the band the model already computes (hot-low to cold-high around the nominal pull-out). No new accuracy claim is made. | A statement of how far the model is from a real coupling needs evidence the project does not have yet (BL-065). |
| C-17 | Torsional stiffness (from item 11) ships with the twist in slice C, as one new engine result with its own physics review. | The twist teaches exactly this: a small twist gives a proportional torque. The owner can strike it. |
| C-18 | Minimum supported window is 1024x700. Phones and a light theme are out. | The tool is a desktop instrument; today's app is dark only. |
| C-19 | O-8 is read as: the given facts stay (the 0.6 N·m gearbox input rating at 5:1, the 43 mm space claim, B842SH magnets at their stock length, the spreadsheet's sleeve, liner, allowances and assumptions), and the requirement becomes what a passing design inside them carries. Required torque is 1.35 N·m: 0.35 N·m of hot bearing drag and 1.00 N·m for acceleration. | The other reading (the spreadsheet's geometry stays and only the requirement drops) cannot pass: three of its four failures do not depend on the requirement (section 7.1). **The owner confirmed the number on 2026-11-10 (O-9). The reading and the design's other five inputs stay this design's call.** |
| C-20 | The opening design is named "Starting design". "Example design" keeps meaning the spreadsheet's design. | About 17 reviewed sentences say "example design" for the spreadsheet's design (for one, "fields stored from the example design's 3D model"). They stay true word for word. No engine text is reworded; one hover loses the word "starting" (section 7.1). |
| C-21 | The baseline is fixed for the session and handed to the panel by the app. `MagcouplingPanel::new` keeps the spreadsheet's values as its baseline. | One constructor argument; no rule to keep right after an undo, a file or a link; a host or a test that builds the panel the old way sees today's behaviour. |
| C-22 | On screen, "default" means the baseline: the changed dot, a row's reset, the "Default:" line of a tooltip, Reset all and the Equation panel's "changed" tags follow it. The spreadsheet export's changed flag and the assumptions banner keep comparing with the workbook. | On screen a dot means "you changed this". In the export it means "differs from the workbook". |
| C-23 | **Share** always copies the full link, at the starting design too. Only the address bar is clean there. | A pasted link must keep its meaning if the starting design changes later (decision M41-5). |
| C-24 | A reload does not keep the "Minimum OD" preference of the Size tab. | It is a view preference, not a design input, and carrying it needs a new key that today's file reader refuses (`session.rs:355-359`). Filed in section 12.4. |
| C-25 | The last click owns the Explain panel. A selection made elsewhere never changes which equation is open (section 5.5). | Two panels that re-point each other are the clunk this work removes. |
| C-26 | BL-068 merges before slice B starts. Slice A does not wait for it. | The Hot-low tile's "needs" line uses BL-068's quoting rule (section 4.4), and O-7 already puts the fixes first. |
| C-27 | The starting design leaves "Torque the coupling must carry for driving" at the spreadsheet's 0.7 N·m. | The owner's bearing figures would make it 0.5 N·m. It judges nothing here either way: the floor is the larger of that torque times 1.3 and Required torque, so 1.35 decides (E, run at 0.5 and at 0.7). One input fewer differs. |

### 1.4 Owner questions (open)

| Id | Question | Recommendation |
|---|---|---|
| Q-1 | The calculator gets its own theme, so decision M41-3 ("Theme matches the linkage app") is reversed and its parity test is replaced. Agreed? | Yes. It follows from O-1 and O-4; it is listed because it reverses a recorded decision. |
| Q-2 | **Answered 2026-11-10 (O-9): 1.35 N·m.** The question was whether to take 1.35 N·m (C-19), on a design with 7 % of margin and a 0.55 mm cap skirt, when the most the given facts allow is 1.69 N·m, on a design with a 0.15 mm cap skirt and 0.6 % of gearbox margin (section 7.1). | Closed. |
| Q-3 | Item 7 names "wheel speed" as optional. C-15 keeps slip heating always judged. Which? | C-15. |
| Q-4 | Release A and B separately, or hold A until B is ready? | Separately. A is small and useful on its own. |
| Q-5 | A link is about 2,700 characters because it names every input (decision M41-5). Keep that, or add a short form later? Section 7.2. | Keep it for slice A; file the short form as a backlog item and decide when someone is bitten by it. |

By O-10, Q-1, Q-3, Q-4 and Q-5 stand on their recommendations for now. Each is an assumption, not the owner's answer, and stays open for the owner to overrule.

## 2. Goals, non-goals, anti-goals

### 2.1 Goals

- **G-1 Fewer words at first paint.** At 1280x800 the opening screen draws at most 250 words, from 727 today (I).
  A test built from the inventory's harness pins it (slice B).
- **G-2 The purpose is visible.** The opening screen shows the drawing with its parts named, four numbers, the
  checks summary and one orientation paragraph, and nothing else that needs reading. The owner judges this by eye;
  no test can.
- **G-3 One place for explanation.** Every input, result, check, dimension and drawing part explains itself in the
  Explain panel, one click away, in the same six-heading form.
- **G-4 Nothing lost.** Every action, input, result, note, plot and export of the inventory is still reachable. A
  table-driven test pins it (section 8.2).
- **G-5 The design survives.** A reload, a bookmark or a pasted address brings back the design being worked on.
  One thing is not kept: the Minimum OD preference, which is not part of a design (C-24).
- **G-6 The first design a visitor sees works.** The starting design passes every check (section 7.1).
- **G-7 One look.** One token table, one font, tabular digits, state by shape and word as well as colour, text
  contrast of at least 4.5:1.
- **G-8 A failing check is one click from its cure.** From the opening screen: one click on the check shows the
  problem, the reason and the fix. From slice D: one more click applies it.

### 2.2 Non-goals

- The linkage tool, the tools hub (but one word: section 9.2) and the landing page (its own to-do).
- New physics. Items 8, 9 and 11 of the list are separate work, except C-17.
- Bringing back the shaft-clamp section. It stays paused (`engine::clamps::PAUSED`), and no on-screen text names it.
- A phone layout, a light theme, translation, accounts, saved-design lists.
- Changing the panel when it is hosted by another egui app: the panel still takes its colours from the host's
  `Visuals` (section 6.3).

### 2.3 Anti-goals (the work fails if any of these is true, even if everything "works")

- A feature that exists today cannot be reached.
- An engine number moves. Slices A, B, E and F change no result of `compute_all`; the parity gates stay green with
  no fixture re-blessed. C adds one result (C-17) and D adds none.
- A failing check's Problem and Fix take more clicks to reach than today (today: zero for a red check, which is
  drawn open; after: one).
- The screen looks clean by hiding what the model cannot do. The "fixed 3D fields from the example design" notice,
  the "Exploration only" lines, the end-effect banner and the assumptions banner all stay visible when they apply.
- A design looks verified because a check is hidden or switched off. The status line always says what is off.
- Text claims what the model does not support. In particular: nothing says the coupling protects the gearbox (the
  test `no_note_says_the_coupling_protects_the_gearbox` guards the notes; section 8.3 extends the guard), and nothing
  says the rings are sealed from each other by a wall. **The mockup's "sealed wall / nothing passes through it"
  line was mine and is wrong; it is not carried over.** The words to use are the notes': two rings of magnets pass
  torque across a gap without touching, and slip past the pull-out torque.
- Decoration that is not information: gradients, glows, motion that is not the answer to the user's own action.

## 3. Slices and order

| Slice | Name | What ships | Depends on |
|---|---|---|---|
| A | First arrival, part 1 | The starting design (section 7.1) and the design in the address bar (section 7.2), on today's screen. | Nothing open (O-9 gives the number) |
| B | The instrument | Theme and font (6), the layout and top bar (4), the Explain panel with its orientation (5), the check list. | A; BL-068 merged (C-26) |
| C | The drawing teaches | Part labels at first paint, hover and click on parts, the twist with its plot marker, torsional stiffness (10.1). | B |
| D | Checks that act | The fix button, the optional application sections, the torque band on the tile (10.2). | B |
| E | The worked example | A five-step worked example replaces Start here (10.3). | C, D |
| F | Lessons | Lessons as web pages generated from the notes; notes for the five rows that have none (10.4). | B |

A and B are specified here. C to F are described well enough to size and order them; each gets a short design
addendum (owner-approved) before its plan, because each holds decisions that are cheaper to make with B on screen.

The owner's first arrival is complete only after C (the drawing's part labels are C's). B and C can be released
together if the owner wants the public first arrival to change once; they are still two plans.

## 4. The screen (slice B)

### 4.1 Layout at 1280x800

```
+------------------------------------------------------------------------------------------------+
| (mark) Magnetic coupling calculator     Undo  Redo  |  Design v   Share   Export  |  Learn     |
+------------------+----------------------------------------------------+------------------------+
| SIZE             | Size and torque | Geometry | Plots | Results       | CHECKS                 |
| Required torque  |                                                    | Starting design: ...   |
| Auto             |  1.70 N·m    1.45 N·m    2.42 N·m     169 g        | (o) Hot-low torque     |
| Minimum OD       |  Pull-out    Hot-low     Cold-high    Rotating     | (o) Running clearance  |
| Outside diameter |  torque      needs 1.35               mass         | (o) Cup wall  ...      |
| Overall length   |                                                    +------------------------+
| Magnet face gap  |   +--------------------+ +---------------------+   | EXPLAIN                |
| Magnets per ring |   |      end view      | |      side view      |   | (orientation, or the   |
| > Detailed inputs|   +--------------------+ +---------------------+   |  selected thing)       |
| > Assumptions    |                                                    |                        |
+------------------+----------------------------------------------------+------------------------+
```

The three columns of today stay (`panel.rs:704-740`: a top bar, a left and a right `SidePanel`, a `CentralPanel`),
so the frame of the panel and most of its position tests survive. The numbers in the sketch are the starting
design's (E, section 7.1) and show the form only.

### 4.2 Top bar

One row. Left to right: the brand mark and the title; Undo, Redo; **Design** (a menu); **Share**; **Export**;
**Learn**. The status line for file and link results keeps its place at the right end.

| Today (`header_ui`, `panel.rs:1226-1331`) | After |
|---|---|
| Undo, Redo | Unchanged. |
| Reset all | Design menu: "Reset to the starting design" (section 7.1). |
| Save design, Load design | Design menu, same words. |
| Copy share link | **Share**. Same action: it copies the full link of the current design (C-23, section 7.2). |
| Export spreadsheet | **Export**. Same action. The Results tab keeps its CSV, JSON and XLSX buttons. |
| Equation panel toggle | Gone from the bar. Opens from the Explain panel's "Equation" link (C-11). |
| Show spreadsheet details | Design menu, a checkbox (C-12). |
| Second row: assumptions banner, Reset assumptions | Unchanged: a thin row under the bar whenever an assumption differs. |
| (none) | **Learn**: lists the shown notes by title in the Explain panel; a title opens the note there. Slice F turns it into the link to the lesson pages. |

The Design menu also holds "Open the example design" (the spreadsheet's values), which slice A first adds to
today's top bar as a button (section 7.1).

### 4.3 Left column: Size

The size controls of today's Size tab (`gui/size_controls.rs`), one row each: label, value with unit, slider.
Required torque joins them as the first row; today it is a slider in the centre (`torque_readouts_ui`,
`panel.rs:948`). Under them, two folds, both closed at first paint: **Detailed inputs** (today's fold, unchanged
inside) and **Assumptions** (today's Assumptions view, which the Size tab cannot reach; inventory section 1b).

What leaves the column: every hint line, "Set by" line, floor line and Why link under a control (145 words, I) and
the `LINKED_HOUSING` footnote. None is deleted. Each shows in the Explain panel when its control is selected, which
happens the moment the control is touched (section 5.2). So a user dragging Magnets per ring still reads "Smallest
outside diameter that fits 10 magnets: 42.1 mm", beside the drawing instead of under the slider.

The left column no longer changes with the centre tab. Today it swaps between the size controls and the workflow
groups; after, the workflow groups live only inside Detailed inputs.

### 4.4 Centre

Tabs: **Size and torque | Geometry | Plots | Results** (C-9). The Clamp tab stays filtered out while the section is
paused.

- **Size and torque** (the opening tab): four tiles, then the conditional "Exploration only" lines (red or amber,
  only when true, unchanged words), then the drawing at the full remaining height. The callout list and the notes
  under the drawing move to the Geometry tab; the six numbered dimension tags stay on the drawing and a click on one
  selects it (its text shows in the Explain panel).
- **Tiles.** Pull-out torque (nominal); Hot-low torque with "needs <required>" under it and the existing gap text
  ("short 0.21 N·m (-8.6 %)" or the margin); Cold-high torque; Rotating mass. The number is large and leads; the
  label is small and muted under it. Each tile is a readout: hover gives today's tooltip and equation, a click
  selects it. A failing hot-low tile is marked by the fail shape and red on its "needs" line, never on the number
  alone. The tile and its "needs" value use the existing formatters, and the pair must go through the same quoting
  rule BL-068 gives the hot-minimum Problem, so a failing tile can never show two equal numbers. BL-068 merges
  before slice B starts (C-26).
  The no-prediction texts (`NO_PREDICTION`, `NO_TORQUE`) replace the number as today.
- **Geometry.** The drawing with the callout list and notes, as today.
- **Plots.** A picker of the five plots (same names, same order), then the plot with its readout line, as today.
- **Results.** The results table, unchanged.
- The welcome strip (`WELCOME`, `panel.rs:76`) is removed; the orientation replaces it. The end-effect banner
  stays above the view when it applies.

### 4.5 Right column: Checks, then Explain

**Checks** (top). The status line, reused word for word (`design_status_text`, `guidance.rs:1953`), then one row per
counted guide in `CHECK_GUIDES` order (C-3): the state shape (C-4), the guide's label, and the check's own quantity
in short form where it has one. A row is a readout (hover as today) and a click selects the check. When a material
warning fires, the "Material warnings" heading and its rows follow, as today. The weak line "Fixed 3D fields from
the example design" closes the list, with its hover.

The list takes its natural height up to about 45 % of the column and scrolls beyond that; the Explain panel takes
the rest and scrolls on its own.

Where today's 19 dashboard rows go: four are the tiles; the rows that carry a check are the check rows; every row's
value is in the Results tab's Headline group (which already holds them) and in the Explain entry of the check it
belongs to ("Numbers behind this check"). The guide blocks (Problem, Why, Fix) are no longer drawn in the list; they
are the Explain entry of the check.

## 5. The Explain panel (slice B)

### 5.1 One selection

The panel holds one `Selection`: nothing, an input, a result (a tile, a dimension, a table row), a check, a note, the
note list, or (slice C) a drawing part. The Explain panel draws it. Selecting replaces the last selection; Esc or the
panel's "About this tool" link returns to nothing, which draws the orientation.

The selection is view state: not in undo, not in a design file, not in the address bar.

### 5.2 What selects

- A click on a tile, a check row, a dimension tag, a results-table row or a callout row selects that result or check.
- Touching an input (a click on its label, a drag of its slider, focus in its field) selects that input. Today a
  click on a label traces the input; tracing stays, as a link in the input's Explain entry ("Highlight the results
  this drives").
- A link inside the Explain panel selects what it names: a fix link selects and focuses its input (as `fix_text`
  links do today, opening Detailed inputs when the input lives there), a note link selects the note.
- Hover selects nothing (C-2). Hover still shows today's tooltips and equation-term marks.

### 5.3 One form

Every entry uses the same headings in the same order, and leaves out a heading it has nothing for.

| Heading | Input | Result or tile | Check | Note |
|---|---|---|---|---|
| Title, with the live value leading | label, value | label, value | label, state | title |
| What it is | `HELP` text | the result's help | the guide's `why` | the note's sentences |
| What sets it | the control's live lines (its hint, "Set by", floors, linked housing) | (omitted; the Equation link shows it) | the guide's live Problem | (omitted) |
| What to change | (omitted) | (omitted) | the guide's fixes, as links | the note's change entries |
| What a change does | its torque-card row (the four size controls) | the whole torque card (Hot-low tile only) | (slice D: the fix preview) | (omitted) |
| Checks it touches | the guides whose fixes name it | the check of its row | the other rows of the check, with values | (omitted) |
| Watch out | the "pulls two ways" sentence | the stored-3D caution where it applies | the conflict sentence | the note's watch-outs |
| Foot links | Full lesson, Equation, Spreadsheet cell | same | same | Where this comes from |

A control's hint is one string with no separate caution part (`gui/plain_text.rs:772`), so the whole hint shows
under "What sets it".

"Full lesson" opens the note the row's `Learn` names (`ROW_LEARN`, `CHECK_GUIDES`), in the Explain panel; "no note
yet" stays the honest text for the five rows without one until slice F. "Equation" opens the docked Equation panel
at that result. "Spreadsheet cell" shows only with spreadsheet details on.

No new explanatory text is written for this table in slice B: every cell reads text that ships today and has passed
its review. The new strings of slice B are the headings, the orientation, the menu and bar words, and the state
words (section 8.3).

### 5.4 The orientation (the panel's first state)

At most 60 words: what the thing is, then the three moves. Draft, for the text review:

> Two rings of magnets, one inside the other, pass torque across the gap between them without touching. Up to the
> pull-out torque the rings turn together; past it they slip.
>
> 1. Set the size on the left.
> 2. Read the torque above the drawing.
> 3. Open any check that needs attention.

Until slice E it ends with a link to the existing Start-here walk, which today can only be started from inside the
Equation panel.

### 5.5 The Explain panel and the Equation panel (C-25)

Two panels can each show "the thing being looked at". One rule keeps them from fighting: the Explain panel shows
the last thing clicked, and the Equation panel changes only through its own controls or an "Equation" link.

- An entry's "Equation" link opens the docked panel at that result. The selection does not change.
- A selection made anywhere else (a tile, a check, an input) does not re-point the open equation.
- While the Start-here walk runs, each step selects its note, so the note shows in the Explain panel. This is what
  the "Explain" checkbox of C-11 did; the walk switches it on today (`gui/explorer.rs`).
- Closing the Equation panel leaves the selection as it is.
- What a click inside the Equation panel selects is left to the builder (section 11), within this rule.

## 6. Look (slice B)

### 6.1 Tokens

| Token | Value | Role |
|---|---|---|
| bg | `#121417` | Behind the drawing and plots; field backgrounds. |
| panel | `#181b1f` | The three columns and the top bar; the web page behind the canvas. |
| panel2 | `#1e2227` | Tiles, menus, tooltips, the selected row. |
| border | `#262b31` | Hairlines, 1 px. |
| text | `#e6e8eb` | Values, headings, body text. |
| muted | `#8a919a` (C-14; the mockup had `#7c838c`) | Labels and captions. |
| accent | `#c8102e` | Brand mark, the active tab's underline, the one primary button (C-5). |
| bad | `#ff4d57` | A failing check. |
| warn | `#e0a100` | A caution. |
| ok | `#4cc38a` | A passing check. |

Contrast, by the WCAG formula (H; the test of section 8.1 recomputes every pair): muted `#7c838c` on panel 4.5:1
and on panel2 4.2:1; muted `#8a919a` on panel 5.4:1 and on panel2 5.0:1; bad on panel 5.3:1; white on accent 5.9:1;
accent as text on panel 2.9:1, which is why C-5 forbids it.

### 6.2 Font

Inter (SIL Open Font Licence), bundled with `include_bytes!` and installed with `FontDefinitions` at app start,
ahead of egui's built-in fonts, which stay as the fallback for any glyph outside the subset.

- **Tabular digits.** egui cannot switch OpenType features, so the font file itself must carry tabular digits as
  its default digits. The font is built once, offline: subset to the characters the app draws (Latin, Greek, the
  maths and unit signs in use) with the `tnum` feature frozen in. The built file, its licence and the exact
  command that made it are committed; the app build does not run font tools. (Built 2026-11-18,
  BL-079: the ten digits alone are remapped to their tabular forms. The whole feature also
  substitutes the space, the full stop, the comma, the colon, the hyphen and the brackets, which
  would widen them in running text. Since the fix wave of that stage's review, 2026-11-22, the
  built file also carries the ascender, descender and line gap of egui's built-in font, so a row
  keeps the height the layout was built on.)
- **Weights.** egui has no synthetic bold (`strong()` changes colour only). One face (Regular) is the default plan;
  a second (SemiBold, for tile numbers and headings) is added only if the spike shows the tiles need it.
- **Cost.** About 0.1 MB per face after subsetting, against a 6.66 MB wasm file today.
- **Spike first.** Before the plan fixes sizes: build the subset, load it in the panel, and check the equations
  (`typeset.rs` and its `glyph_safe` substitutions were tuned to the built-in fonts), the drawing's tags and the
  plots at 1280x800 and 1920x1080. The spike is throwaway; its findings go into the slice B plan.
  (Done 2026-11-17: the findings are in the stage 1 plan, under "What the spike found".)

### 6.3 Where the theme lives

In `app/theme.rs`, as today: the app sets the `Visuals`, the panel reads them. So the panel still takes a host's
theme if it is ever embedded. The few colours the panel owns as constants (the Good green at `dashboard.rs:88-96`,
the drawing's and plots' palettes) take the token values where a token exists and are otherwise unchanged; the
drawing's part colours are not changed (C-7).

The parity test `the_visuals_are_the_linkage_app_s` is replaced by a test that the `Visuals` equal the token table,
and `the_web_page_background_is_the_panel_colour` moves to the new panel colour in
`linkage-sim-rs/web/tools/magcoupler/index.html` (Q-1).

## 7. First arrival, part 1 (slice A)

### 7.1 The starting design

**Finding.** At the spreadsheet's values the tool opens on "Example design: fails 4 of 9 checks" (hot minimum,
clearance, cup wall, gearbox). A visitor's first sight of the tool is four red blocks.

**The owner's ruling (O-8).** The machine is "wheel, wheel bearings, mag coupler, 5:1 gearbox, belt, 1:1 gearbox,
motor", and the motor drives the wheel. "Wheel bearing are around 0.5 Nm cold, or 0.35 Nm hot. Acceleration torque
is as much as we can fit in this space, update the requirements to match what can be achieved with the tested/given
default parameters".

**How this design reads it (C-19, mine).** "The tested/given default parameters" can be read two ways.

- *The spreadsheet's geometry stays and only the requirement drops.* This gives no passing design. With Required
  torque at 2.25 N·m, just under the 2.285 N·m the spreadsheet's design carries at hot-low, it still fails 3 of 9
  checks (E): running clearance (-0.10 mm against a 0.20 mm target), cup wall (1.80 mm against the 1.92 mm its flux
  needs) and gearbox input torque (0.80 against 0.60 N·m). None of the three depends on the requirement.
- *The given facts stay and a design is sized inside them.* The facts are the 0.6 N·m gearbox input rating at 5:1
  (which the owner confirmed on 2026-10-14), the 43 mm space claim, B842SH magnets at their stock length on both
  rings, and the spreadsheet's sleeve, liner, allowances and assumptions. **This is the reading taken.**

**What the given facts allow** (E; an engine search of 2026-11-09, judged by the GUI's own check functions; section
12.4 says where the probe is).

- **The gearbox rating, not the space, sets the ceiling.** The gearbox check allows at most 2.85 N·m of cold-high
  at the coupling (the 0.6 N·m input rating times the 5:1 ratio and its 0.95 efficiency). In the model, cold-high
  is a fixed multiple of hot-low, which the magnet grade sets and the geometry does not move: 1.67 for every NdFeB
  library part, 1.44 for SmCo 2:17, 1.47 for SmCo 1:5, 1.73 for bonded NdFeB and 1.92 for ferrite (the 15 %
  variation allowance alone gives 1.35). So with NdFeB magnets hot-low can be at most 1.70 N·m, whatever the size.
  Of 462,528 designs (15 library parts, 17 grades, 4 to 40 magnets, radius 9 to 30 mm, gap 0.3 to 5 mm, four
  lengths) none passes both the gearbox check and a hot-low requirement of 2.5 N·m.
- **The space comes second.** Only one library part passes the temperature check and fits the 20 mm bay (B842SH).
  Over 659,340 geometries of it, the best hot-low inside the 43 mm diameter claim with the clearance and cup-wall
  checks passing is 2.09 N·m. 2.5 N·m would take a 45.3 mm cup, so a claim of about 47 mm.
- **Inside both, with stock-length magnets, hot-low tops out at 1.69 N·m**, on a design that could not be made (the
  right-hand column below).

| | Starting design (taken) | Ceiling design (not taken) |
|---|---|---|
| Magnets per ring | 8 | 10 |
| Ring radius | 9.00 mm | 9.85 mm |
| Magnet face gap | 1.80 mm | 2.25 mm |
| Minimum outer return ring wall | 2.20 mm | 1.80 mm |
| Cap thread nominal diameter | 41.7 mm | 42.5 mm |
| Hot-low torque | 1.448 N·m | 1.693 N·m |
| Cold-high at the gearbox input | 0.510 N·m, 15 % under the rating | 0.596 N·m, 0.6 % under the rating |
| Cap skirt (cap outside diameter less thread, halved) | 0.55 mm | 0.15 mm |
| Passes all 9 checks up to a Required torque of | 1.44 N·m | 1.69 N·m |

Between the two the search found nothing worth taking instead. An eight-magnet design at radius 9.05 mm and gap
1.75 mm reaches 1.47 N·m with the same skirt but 0.01 mm of clearance margin (E), and no stock-length design has a
skirt of 0.70 mm or more. Magnets cut to a custom length reach 1.60 to 1.70 N·m with skirts of 0.55 to 0.70 mm,
but they are not library parts.

**The requirement (C-19, O-9).** Required torque is judged against hot-low, so it is the hot case written as one number.

- Hot: hot-low 1.448 N·m less 0.35 N·m of bearing drag leaves 1.10 N·m to accelerate the wheel.
- Cold: the cold torque on the low side of the variation allowance is 1.79 N·m (H: cold-high 2.423 divided by 1.15,
  times 0.85). Less 0.50 N·m of bearing drag it leaves 1.29 N·m. So the hot case decides.
- **Required torque = 0.35 + 1.00 = 1.35 N·m.** The design then shows 7.3 % of margin on the hot-low check. 1.40
  would show 3.4 % and 1.44 would show 0.6 %; all three pass all 9 checks (E).

The tool has no separate inputs for bearing drag and acceleration torque, so on screen there is one number. The
split is written in this document, in the README and in a comment beside the starting design's values; section
12.4 files the gap. "Torque the coupling must carry for driving" stays at 0.7 N·m (C-27).

This design changes no check. The gearbox check has no input for which side drives; it judges cold-high at the
gearbox input, as today.

**The starting design's rules.**
- It passes every counted check with no caution, and no material warning fires. A test pins this to `compute_all`.
- It changes no assumption (the assumptions banner stays quiet) and uses only library parts and grades, at their
  stock length.
- It differs from the spreadsheet's values in six inputs, and every differing value is one a slider step can
  reach, so a user can walk from one to the other. (Amended 2026-11-22, BL-080, by the owner's rulings on register
  rows A-17 and A-18: a seventh input, the keyway depth in the hub, 1.4 mm against 1.7 mm, and its fourteen stored
  3D fields, computed for its own geometry; see the end of this section.)

| Input (on-screen label) | Spreadsheet | Starting design |
|---|---|---|
| Magnets per ring | 10 | 8 |
| Ring radius | 10.15 mm | 9.00 mm |
| Magnet face gap | 1.40 mm | 1.80 mm |
| Minimum outer return ring wall | 1.80 mm | 2.20 mm |
| Cap thread nominal diameter | 41.0 mm | 41.7 mm |
| Required torque | 2.50 N·m | 1.35 N·m |
| Keyway depth in the hub (amended 2026-11-22, BL-080) | 1.70 mm | 1.40 mm |

The result (E; re-run on the real engine on 2026-11-09: "passes all 9 checks", no caution, no warning, no drawing
note):

| Check | Result |
|---|---|
| Hot-low torque | 1.448 against 1.35 N·m (+7.3 %) |
| Running clearance | 0.263 against 0.20 mm |
| Cup wall | 2.200 against 2.126 mm needed |
| Temperature limits | 28.1 °C hot-day margin |
| Shaft joints | holds 8.64 against 4.85 N·m |
| Gearbox input torque | 0.510 against 0.60 N·m (15 % under) |
| Space claim | 0.2 mm on diameter, 3.2 mm on length, 1.2 mm in the bay |
| End-effect model | factor 0.879 |
| Magnets fit | 1.11 mm of flat slack |

Pull-out at 50 °C is 1.70 N·m, cold-high 2.42 N·m, rotating mass 169 g, outside diameter 42.8 mm, length 31.8 mm.
The magnets stay B842SH on both rings at the stock length.

**Two named designs, one baseline** (C-20, C-21 and C-22, all mine).

- **Starting design.** What the app opens on when the address carries no `?m=`. The status line reads "Starting
  design: passes all 9 checks". The name and its hover are new strings and take the text review. Draft hover: "A
  design that passes every check. It keeps the example design's gearbox rating, space and magnet part, and asks for
  less torque." (The text review of 2026-11-17 changed "magnets" to "magnet part": the part is kept, the number
  per ring is not; `docs/analyses/2026-11-17-magcoupler-slice-a-review.md`.)
- **Example design.** The spreadsheet's values, `DesignInputs::default`, unchanged. Its name and every sentence
  that says "example design" today stay as they are, and stay true: the stored 3D fields do come from that design.
  (Amended 2026-11-22, BL-080: the starting design now stores 3D fields of its own, so the texts about the stored
  fields name both: "Fixed 3D fields: starting or example design".)
  One string changes: its hover, "The starting values from the original spreadsheet." (`EXAMPLE_DESIGN_HOVER`,
  `gui/guidance.rs`), loses the word "starting".
- **Reaching the example design.** Slice A adds one button to today's top bar, "Open the example design"; slice B
  moves it into the Design menu. Opening it is one undo step, as loading a file is. Without it the spreadsheet's
  design could no longer be reached from the screen, because Reset all now goes to the baseline.
- **The baseline.** The panel holds one baseline design, set when the panel is built and fixed for the session.
  The app hands it the starting design through one new constructor. `MagcouplingPanel::new` keeps the
  spreadsheet's values as its baseline, so a host or a test that builds the panel that way sees today's behaviour.
  Nothing moves the baseline afterwards: not an undo, a file, a link or the example design. It is not in the undo
  history and is not saved. The starting design itself is `DesignInputs::default()` with six values set (amended 2026-11-22, BL-080: twenty-one values, the seven it sizes and its fourteen stored 3D fields), written
  once in the app's code; where exactly is the plan's.
- **What follows the baseline:** the changed dot, a row's reset, the "Default:" line of an input's tooltip, Reset
  all (slice B words it "Reset to the starting design") and the Equation panel's "changed" tags. Their words do not
  change; "the default" on screen is the baseline's value.
- **What stays against the workbook:** the spreadsheet export's changed flag (`gui/spreadsheet.rs`), and the
  assumptions banner with the dots of the Assumptions view. The starting design changes no assumption, which test
  1 pins, so the two baselines agree there.
- **The status word** is "Starting design" while the design equals the baseline and the baseline is not the
  spreadsheet's values, "Example design" while the design equals the spreadsheet's values (as today), and "Your
  design" otherwise.
- At the example design the six inputs (amended 2026-11-22, BL-080: twenty-one inputs, the seven it sizes and its fourteen stored 3D fields) that differ from the starting design carry the changed dot. That is meant:
  the dots show what the starting design changed.
- A link or file made before this change opens to the same design as before. Only the address with no `?m=`
  changes what it opens. A link that carries the spreadsheet's values opens as "Example design".

**What the owner should know about the starting design.**
- **The cap's skirt is 0.55 mm.** (Cap OD 42.8 less thread 41.7, halved.) No check judges that wall
  (`engine/metal_design.rs:349-351` takes both diameters as given), so the tool shows no caution. The spreadsheet
  design's is 0.90 mm on paper, with its thread below the cup body, which the drawing flags today. A thicker skirt
  needs a larger diameter claim. Section 12.4 files the missing check.
- **Ring radius 9.00 mm is the slider's minimum.** A user cannot walk further down from it.
- **The opening tab drew one amber line; answered 2026-11-22** (found by the slice A plan, 2026-11-17; register row A-17). The owner chose the standard keyway depth: the starting design's hub keyway is 1.4 mm (DIN 6885-1, t2 for a 3 x 3 mm key on a shaft over 8 up to 10 mm), the wall past it 2.55 mm, and the line is not drawn; every check keeps its verdict and margin (BL-080). As found: At a ring radius of 9.00 mm the hub wall past the keyway is 2.25 mm, and the Size tab says so against its guidance of about 2.5 mm ("Wall past keyway: 2.25 mm; existing guidance is about 2.5 mm."). No counted check judges it, so the status line still reads "passes all 9 checks". A ring radius of 9.26 mm clears the line and takes the cap's skirt from 0.55 to 0.30 mm (with the cap thread at 42.2 mm; at 9.25 mm the wall computes a hair under 2.5 mm and the line still shows). The approved values are kept and the line stays visible until the owner rules.
- **Two of its nine checks used the example design's 3D fields; answered 2026-11-22** (found by the slice A review, 2026-11-17; register row A-18). The owner chose to compute them: the reference routine was run once for the starting design's geometry and its fourteen values are stored with it (`gui::dashboard::STARTING_3D_FIELDS`, BL-080). It still passes all 9 checks with no caution; the magnet limit goes from 93.06 to 94.94 °C and the inner hub's fitting margin stays 6.73 °C. The fields stay fixed under an edit, and the dashboard line reads "Fixed 3D fields: starting or example design". As found: The temperature check, and the inner hub's fitting temperature inside the shaft joints check, read the stored reverse fields, which were computed in 3D for the example design's geometry and stay fixed for every design. Until slice A the page opened on the design those fields belong to. The dashboard says so under those checks ("Fixed 3D fields from the example design", with its hover); at 1280x800 that line is below the dashboard's first screen. With the four stored reverse fields raised by 10 % the starting design still passes all 9 checks with no caution; raised by 25 % the temperature check turns to a caution, and raised by 75 % the shaft joints check fails (probe of 2026-11-17, `docs/analyses/2026-11-17-magcoupler-slice-a-review.md`). Nothing is changed until the owner rules.
- **1.35 N·m is a choice inside a range, and the owner took it (O-9).** Anything up to 1.44 N·m passes on this design. The ceiling
  design's 1.69 N·m is the most the given facts allow; it is not taken because of its 0.15 mm skirt and its 0.6 %
  of gearbox margin.
- **The numbers are at 1048962.** A fix that merges first (section 9.3) can move them. The plan re-runs the check
  on the base it branches from and adjusts the design if a margin is gone; test 1 then holds it.

**A finding about the real coupling, outside this design.** With the spreadsheet's 2.5 N·m, the 0.6 N·m gearbox
rating and the 43 mm space, the model has no passing design; the tool states that today only as four red checks.
With NdFeB magnets the gearbox rating holds hot-low to 1.70 N·m. Less the hot bearing drag that leaves at most
1.35 N·m for acceleration, with no margin (equal to the starting design's Required torque only by chance; that
one is 0.35 + 1.00). More than that takes what the gearbox check's own Problem text names: a
gearbox rated for more, or a narrower band between hot-low and cold-high (a warmer minimum temperature, SmCo
magnets, or a smaller variation allowance backed by measurement).

**Before O-8.** The first draft of this section offered four options: lower the requirement; raise the gearbox
rating and the space claim; open on one failing check; or switch the gearbox section off. O-8 chose the first, so
the other three are dropped.

### 7.2 The design in the address bar

- After every settled edit (the moment an undo step is recorded: `gui/history.rs`), and after undo, redo, a file
  load or "Open the example design", the web app rewrites the page's address to the share link of the current
  design, with `history.replaceState`. The Back button therefore still leaves the tool; it does not walk edits.
  Undo is the tool's.
- At the starting design the address is the clean one, with no `?m=`. **Share** still copies the full link there
  (C-23), so a pasted link names every input.
- The link is the one **Share** copies today (`share_link`, `session.rs:448`; payload from `encode_share_payload`,
  `session.rs:408`; `SHARE_PARAM = "m"`). No second encoding.
- A link that cannot be read changes nothing, says why in the status line, and leaves the address as it was until
  the first edit.
- Browsers limit `replaceState` (Safari refuses more than about 100 calls in 10 seconds). The app writes only a
  settled design, and never while a key is held: a held Ctrl+Z or Ctrl+Y undoes or redoes one step at every key
  repeat, each step a settled design, so the whole run gives one write when the key is released. A refused write
  leaves the address standing for the last design it took, and the app asks again on its next frame. (Changed by
  the independent review of slice A, 2026-11-17: the first build wrote at every step of a held undo and counted a
  refused write as written.)
- The native app does nothing. The selection, the tab and every other view flag stay out of the address.
- One thing a reload does not bring back: the Minimum OD preference of the Size tab (C-24). It is view state
  (`panel.rs:387-388`), not a design input, and the file reader refuses a key it does not know
  (`session.rs:355-359`). Section 12.4 files it.
- Code: `web-sys` gains the `History` feature (`M/Cargo.toml:93`); the write sits beside the read in
  `src/bin/magcoupling_web.rs:33-44` or in `MagcouplingApp::update` (`src/app.rs:104`), behind
  `cfg(target_arch = "wasm32")`. Two hooks exist: `MagcouplingPanel::design` (`panel.rs:474`) and `share_link`
  (`panel.rs:566`). The third does not: the settled flag is worked out inside the panel (`panel.rs:771-772`) and
  `History` has no getter for it (`gui/history.rs:32-93`), so the panel gains one small read-only accessor that
  says whether the design is settled. The app keeps the last design it wrote and writes again only when a settled
  design differs from it, so an idle frame encodes nothing.
- **Length.** A link is about 2,700 characters whatever the design (five designs encoded with
  `encode_share_payload` on 2026-11-09: payloads of 2,642 to 2,647 characters, links of 2,687 to 2,692), because
  a design file names every input so that it keeps its meaning if a default changes later (decision M41-5). Today's Share links are already this long; slice A
  keeps the form and adds no second one. Two things follow. The address is long but browsers elide it. And a link
  does not fit in a 2,000-character chat message, which is true of Share today. A short form (only the inputs that
  differ from the spreadsheet's values; the reader already takes a missing input at its default) would cut a
  typical link to a few hundred characters but gives up M41-5 for links; it is not in this design and is filed as
  a backlog item for the owner to rule on (Q-5).
- The plan pins the length with a test (a ceiling of 4,000 characters) and loads one full-length link on the live
  site in the smoke run. The old-host redirects in `linkage-sim-rs/web/vercel.json` pass `?m=` through and need no
  change.

## 8. Tests (red first: each fails before its change and passes after)

### 8.1 Slice A and the look

1. The starting design passes every counted check with no caution and no material warning, pinned to
   `compute_all`. It changes no assumption, and each of its six differing inputs (amended 2026-11-22, BL-080: twenty-one inputs, the seven it sizes and its fourteen stored 3D fields) sits on its slider's step.
2. `DesignInputs::default` is unchanged and the parity gates pass with no fixture re-blessed. A panel built with
   `MagcouplingPanel::new` behaves as today: the tests that read the status word and the default comparisons
   (`guidance.rs:8966`, `8982` and `8986`; `dashboard.rs:2532`; `panel.rs:4175` and `10214-10236`) keep their
   assertions. Where a function gains a baseline argument the call is edited and the assertion is not; the plan
   lists each such call.
3. The app opens the starting design. The status line reads "Starting design: passes all 9 checks"; no changed
   dot is drawn; the Equation panel shows no "changed" tag; the tooltip of a differing input gives the starting
   design's value as its default.
4. Baseline: after "Open the example design" the status reads "Example design: fails 4 of 9 checks" and exactly
   the six differing inputs (amended 2026-11-22, BL-080: twenty-one inputs, the seven it sizes and its fourteen stored 3D fields) carry the dot; a row's reset returns the starting design's value; Reset all returns
   the starting design; the opening is one undo step, and undoing it gives "Starting design" with no dot. The
   spreadsheet export still flags against the workbook: the six differing inputs (amended 2026-11-22, BL-080: twenty-one inputs, the seven it sizes and its fourteen stored 3D fields) at the starting design, none at
   the example design. The assumptions banner is quiet at both.
5. Address: an edit changes the link; the starting design gives the clean address while Share returns a full
   `?m=` link; undo and redo move it; a decoded link equals the design that made it; a link that carries the
   spreadsheet's values opens as "Example design"; a bad payload changes nothing and sets the status line. The
   `replaceState` call itself is thin and is covered by the browser smoke run (the `gui-smoke` skill), which
   reloads the page after an edit and reads the design back.
6. Theme: the `Visuals` equal the token table; the web page background equals the panel colour; every text token
   on every surface it is drawn on is at least 4.5:1, computed by the test.
7. Font: every character of every user-facing string, and every glyph the typesetter emits, has a glyph in the
   font stack (the existing guard `every_text_the_panel_shows_has_glyphs_and_none_names_the_paused_clamp`,
   `panel.rs:4565`, run against the new stack); the ten digits have one advance width.

### 8.2 Slice B

8. **Word budget.** The inventory's harness, turned into a test: the opening screen at 1280x800 draws at most 250
   words.
9. **Nothing lost.** Table-driven: every action of today's top bar is in the bar or the Design menu; every
   `CentreView` is reachable; every shown note is in the Learn list; every `DASHBOARD` row's path is a tile, a
   check row's Explain entry or a Headline row of the results table; every input is in the Size column or Detailed
   inputs; the Assumptions view is reachable from the opening tab.
10. **Selection.** A click on each kind of target puts that target in the Explain panel; a hover changes nothing;
    Esc returns the orientation; the Explain panel shows the orientation on the first frame. A selection made
    elsewhere leaves the open equation where it is, and each step of the Start-here walk selects its note (5.5).
11. **Entries.** For each guide, the check's Explain entry holds exactly the Problem, Why and fixes the dashboard
    block holds today (same strings, same links). For each size control, its entry holds the lines the left column
    shows today and its torque-card row.
12. **State without colour.** Each check row draws a shape that differs by state.
13. **Still honest.** The fixed-3D line, the end-effect banner, the "Exploration only" lines and the assumptions
    banner are drawn when their conditions hold. No drawn text names the paused section (the existing test at
    `panel.rs:4565`, extended to the new surfaces).
14. **Idle frames.** The Explain panel and the check list add no `compute_all` to a frame in which nothing changed
    (the pattern of spec3 section 8.2).
15. **Layout.** At 1024x700, 1280x800 and 1920x1080: no column overlaps another, the drawing keeps its minimum
    height (`MIN_DRAWING_HEIGHT`), the four tiles wrap to two rows instead of clipping.

### 8.3 Text rules (every slice)

- Every new on-screen string goes through the plain-words guard (`BANNED`, `engine/explain/plain.rs:39`) and the
  text review of spec1's decision G-3; a string that states physics also gets a physics review.
- The gearbox-protection guard (`plain.rs::mentions_gearbox_protection`) is extended from the notes to every
  user-facing string. Added to the ban list for new strings: "sealed", "containment", "leak" (the tool does not
  model a pressure boundary), and ring names by colour.
- No "clamp" on screen while the section is paused.
- A number in a sentence comes from the engine, through the existing formatters; no sentence carries a typed-in
  number.

## 9. Docs, backlog, release

### 9.1 With each slice (in the commit that makes it true)

`M/README.md` (the GUI section), `docs/ai/*.yaml` (read first, updated after), `docs/ai/backlog.yaml` (the slice's
item; new findings filed, never fixed in passing), the decisions of section 1 recorded in `docs/ai/04-memory.yaml`,
and the `gui-smoke` checklist.

### 9.2 Small fixes folded into slice B

- The hub card's blurb at `linkage-sim-rs/web/tools/index.html:65` still lists "clamps": the word is removed.
- The slip-heating readout "never: steady state stays below the limit s" loses its stray unit.

### 9.3 The fixes branch

`magcoupling/off-default-fixes` (worktree `C:/Users/Cole/source/repos/lsim-fixes`) carries item 9 one fix at a time
and merges on its own, on the owner's OK. It touches `gui/guidance.rs` text and engine values, not layout. Each
overhaul slice branches from `main` after whatever has merged. Slice A waits for none of the fixes. Slice B starts
only after BL-068 has merged (C-26), because the tile's quoting rule (section 4.4) uses its fix. After each merge
the starting design is run again (test 1).

### 9.4 Release, per slice

Plan (owner-approved) -> build -> text and physics reviews -> whole-branch review -> full gate (`scripts/gate.sh`)
-> local `gui-smoke` -> **stop for the owner's OK** -> merge -> push (which deploys) -> live `gui-smoke`. Model roles
follow CLAUDE.md section 5: reviews, physics and design on `fable`; transcription from an exact plan on `sonnet`;
git housekeeping on `haiku`.

## 10. Later slices (described, not specified)

### 10.1 C: the drawing teaches

- **Part tags.** `Piece` (`gui/geometry.rs`) gains the assembly it belongs to (inner ring, outer ring, cup, hub,
  sleeve, liner, cap, shaft, key). The gap is not a piece; it gets a hit region of its own. A hit test per outline
  kind runs through the inverse of the drawing's `Transform`; a hover draws a thin light outline; a click selects
  the part.
- **Labels.** The parts are named on the drawing at first paint, with leader lines, until the first click on the
  drawing; after that a name shows on hover.
- **Part texts.** One short Explain entry per part: what it is, what it is made of (live), which inputs size it,
  which checks it touches. These are new texts and take both reviews.
- **Twist.** A "Twist" slider under the drawing, and a drag on the inner ring that moves the same value, turn the
  inner ring in the end view. A readout gives the torque at that twist as a number and as a share of pull-out, from
  the engine's Torque vs rotation curve; the same point is marked on that plot. Past the pull-out angle the readout
  says the rings slip to the next pole. Releasing the drag returns the ring to zero. The twist is view state.
- **Torsional stiffness** (C-17): the slope of that curve at zero twist, as one engine result with a record, tests
  and a physics review.
- The addendum settles: the angle range and sign, what is drawn past pull-out, and the part texts.

### 10.2 D: checks that act

- **Fix button.** In a failing check's Explain entry, each fix that can cure the check alone becomes a button: "Set
  <label> to <value>". The value is the first slider step in the fix's direction at which that check passes, found
  by bisection over the slider's steps and confirmed by a real `compute_all` (a fix that does not confirm is not
  offered); for hot-low the existing solver is used (`engine/sizing.rs:236`). Under the button, a preview in the
  torque card's words of every other check the change flips. One click is one undo step. Nothing chains: the tool
  never applies a second fix on its own. A fix with no passing step in range says so and keeps today's link. The
  search runs only for the open check and is cached with the design. It reuses `fixes`, `helps`, `step`,
  `trade_offs` and `side_effects` in `gui/guidance.rs`.
- **Optional sections.** The pause mechanism (`engine::clamps::hides`, `guidance::counts`) is generalised from one
  constant to per-design switches for Gearbox and Shaft joints (C-15). A switched-off section's inputs, rows and
  check are hidden and its check leaves the count; the engine still computes it and the exports keep its values,
  marked as not judged. The status line always names what is off ("passes all 7 checks; not judged: Gearbox, Shaft
  joints"). The switches travel in design files and links: the file reader refuses unknown keys today
  (`session.rs:357`), so the format goes to version 2 and version 1 still loads.
- **Torque band** (C-16): a small range bar on the Pull-out tile from hot-low to cold-high with the nominal marked.
- The addendum settles: the button's exact words, which fixes are "alone" fixes, and the export's wording.

### 10.3 E: the worked example

Five steps on the live tool, replacing the 14-step Start-here walk: what the coupling does (the drawing and the
twist); set the requirement; size it until hot-low passes; read a failing check and apply its fix; where to learn
more. Each step points at one region and waits for the user's own action or Next; Stop works at any step and
restores the user's design. The notes the old walk visited all stay reachable from Learn. The addendum settles the
five steps' words and the design each starts from.

### 10.4 F: lessons

Each reviewed note becomes a web page under `/tools/magcoupler/learn/`, generated from `engine/explain/notes.rs` at
build time so the page and the panel cannot drift, with links that open the calculator at a design (and, with one
new address parameter, at a selection). "Full lesson" and **Learn** then point there. The five rows that say "no
note yet" (Cup outer diameter, Rotating mass, Clearance, Space claim, Magnets fit) get notes in the existing form
(`Note`: 3 to 5 sentences of at most 25 words, 1 to 5 changes, up to 2 watch-outs), each with its physics review
and the owner's sign-off as in slice 2. The addendum settles the hard part: the notes' diagrams and equations are
drawn by egui today (`gui/diagrams.rs`, `gui/typeset.rs`) and need a second renderer for a web page, or pictures
made at build time.

## 11. Left to the builder

- The mapping from tokens to `Visuals` fields, the type sizes (body no smaller than today's 12.5 pt; tile numbers
  20 to 24 pt), spacing, and whether the column widths (`INPUTS_WIDTH` 320, `DASHBOARD_WIDTH` 300) move. Keeping
  them saves most of the position tests; the Explain panel's line length at 300 pt is the thing to check.
- The module split. `panel.rs` and `guidance.rs` are already very large; the new surfaces (selection, Explain
  panel, tiles, check list) should be modules of their own, not more lines there.
- How the selection and the existing trace, focus and Equation-panel state share one owner without two sources of
  truth.
- The tab underline, tile and check-row widgets; the menu widget for Design.
- Short forms of each check's quantity in the check list (from the numbers the guide's Problem already quotes).
- Every exact string, within section 8.3.

## 12. Out of scope, risks, not verified

### 12.1 Out of scope

Section 2.2, and: storage between visits; the selection in the address bar (until slice F needs it); any change to
the exports' contents in slices A to C; the items of the fixes branch.

### 12.2 Risks

- **Test churn in `panel.rs`.** Dozens of tests assert positions and first-paint content. Slice B rewrites many of
  them. The risk is losing a guarantee in the rewrite; the plan lists each test that changes with what replaces it,
  as spec3 section 6.3 did.
- **Words moved, not removed.** The budget counts first paint. If the Explain entries grow without limit the clunk
  moves into the panel. Each heading of section 5.3 shows existing reviewed text and nothing else in slice B.
- **One click more for a red check.** Today a red check's block is open on arrival. After, the starting design has
  no red check, and a failing one costs a click. G-8 and the anti-goal accept exactly one.
- **The font.** Subsetting can drop a glyph the equations need; a frozen feature can change a glyph's width in a
  place tuned by hand. The spike (6.2) and test 7 exist for this.
- **Crimson beside red.** C-4 and C-5 are the mitigation; the owner should look at a failing design in the built
  slice B before release.
- **A starting design that drifts.** A later physics fix can make it fail. Test 1 fails loudly; the cure is a new
  starting design, never a loosened test.
- **Two names.** "Starting design" and "Example design" sit side by side, and a visitor may not tell them apart
  before hovering. The owner should look at both status lines in the built slice A (C-20). Renaming the
  spreadsheet's design is filed in section 12.4.
- **Address length.** Every link is about 2,700 characters (7.2). Some chat tools cut or refuse links that long;
  that is true of Share today and is not made worse, but the address bar makes it more visible (Q-5).

### 12.3 Not verified

- That egui 0.32 draws the frozen-`tnum` Inter subset correctly, and what it costs in wasm bytes (the spike).
  Checked 2026-11-17 in a headless render of the real app: the first view, the Geometry tab's drawing with its
  tags and one plot at 1280x800, and the Equation panel's equation at 1920x1080 (the pictures the stage 1 plan
  lists); the file is 30,684 bytes. A browser's own drawing of the look is the smoke run's `look`, which sees
  the theme's marks there (the charcoal page, the crimson square, the line under the active tab). It cannot tell
  Inter from egui's built-in font: both are one clean sans-serif face to the agent that judges the picture. That
  the app installs Inter is held by the Rust tests (`the_app_draws_its_text_in_inter`), on the code path the web
  build shares.
- That the host serves an address of 2,700 characters (the limit of the Vercel edge was not found in its
  documentation; a Share link of that length has not been loaded on the live site in this work).
- The live site against this checkout: the inventory read the code, not colesorkness.com.
- Hover texts as drawn: the harness could not read them (inventory section 10), so no heading of section 5.3 has
  been checked against a hover's exact words.
- Window sizes other than 1280x800 for any (I) number.
- The contrast figures are hand calculations to two digits.
- The starting design (7.1) in the built web tool: it was judged natively through the GUI's functions, not in the
  wasm build, and its link was not opened on the live site.
- Whether a 0.55 mm cap skirt, or eight outer blocks 5.22 mm apart at their faces, can be made.
- The search behind 7.1 took its candidates from B842SH on both rings. Mixed parts on the two rings and SmCo
  candidates were not tried for the starting design (the two "no design passes" results cover every library part
  and grade). Custom magnet lengths were searched and set aside because they are not library parts.
- The bearing drag figures (0.5 N·m cold, 0.35 N·m hot) are the owner's; nothing in the repository measures them.
  The cold-side 1.79 N·m of section 7.1 is a hand calculation, not an engine result.
- The line references the builder check gave for the tests named in test 2 were not opened again one by one.
- Whether a visitor tells "Starting design" from "Example design" without help (C-20).

### 12.4 Found on the way (not in this design)

The search probe is scratch code outside the repository; it calls `compute_all` and `gui::guidance` on a design
and prints the nine checks. Test 1 replaces it. The work turned up nine things, none checked further. Each is
proposed as a backlog item for the owner to accept or strike; none is fixed in an overhaul slice.

1. Nothing checks the cap's skirt wall (`engine/metal_design.rs:349-351`): a design with a 0.15 mm skirt passes all
   9 checks.
2. The spreadsheet design draws a "Design check:" note today (`gui/geometry.rs:822-833`): its cap thread, 41.00 mm,
   is below the cup body, 41.33 mm. The same note can print two equal numbers when they differ past the second
   decimal.
3. On the Size tab the diameter control carries the negative thread offset along and puts Ring radius off its
   slider step (`engine/size_controls.rs:164-165` and `446-447`).
4. `model.verdict` reads "Nominal only: hot test" on every design that was evaluated, passing ones included.
5. `temperature.magnet_life.torque_hot_day_check` reads "Meets it nominally (no variation allowance)" on a design
   that passes all 9 checks; whether a reader takes that as a caution was not looked at.
6. The Minimum OD preference of the Size tab is view state (`panel.rs:387-388`). A reload, a file and a link all
   lose it (`session.rs:355-359` refuses a key it does not know).
7. The tool takes the requirement as one number. It has no inputs for bearing drag or acceleration torque, and its
   nearest input, "Torque the coupling must carry for driving" (0.7 N·m, times a 1.3 safety factor), does not
   match the owner's bearing figures (0.5 N·m cold, 0.35 N·m hot).
8. "Example design" names a design that fails 4 of 9 checks. Renaming it (for one, "Spreadsheet design") touches
   about 17 reviewed sentences in the engine and the GUI, each needing its text review.
9. A short link form (Q-5, section 7.2).

### 12.5 The builder check (2026-11-09)

A reviewer read this design against the code as the builder who would plan slices A and B. It built and ran
nothing. Where each of its points went:

| What the builder asked or found | Where it is settled |
|---|---|
| Does the panel or the app own the named designs? | The app owns the starting design and hands it over as the baseline (C-21, section 7.1). |
| "Example design" already names the spreadsheet's design in about 17 strings. | The opening design is "Starting design"; those strings stay (C-20). |
| Which other "default" surfaces follow the new design? | C-22, section 7.1. |
| What is the baseline when the design equals no named design, or after an undo? | The baseline is fixed for the session (C-21). |
| Share at the opening design: a clean address or `?m=`? | A clean address, and the full link from Share (C-23, section 7.2). |
| Minimum OD is lost on a reload. | Accepted, with a caveat in G-5 and a backlog item (C-24, section 12.4). |
| Slice B needs BL-068's quoting rule, and BL-068 is still open. | BL-068 merges before slice B starts (C-26, section 9.3). |
| What drives the Explain panel while the Equation panel or the walk is open? | Section 5.5 (C-25). |
| "The panel tests pass untouched" was wrong. | Test 2 now says what holds. |
| The settled flag is not exposed. | One new accessor (section 7.2). |
| A hint has no separate caution part. | Section 5.3. |
| `header_ui` ends at line 1331, not 1334. | Section 4.2. |

Every point was answered from the code or settled as a call of mine (C-20 to C-26). None changes what the owner
already decided; the calls are in the hand-over list for the owner to check.
