export const meta = {
  name: 'gui-smoke',
  description: 'Smoke-test the locally served web build (the hub page at /tools/, the linkage app at /tools/linkage/, the calculator at /tools/magcoupler/): loads, canvas present, console clean; on the calculator the clean address opens the starting design in its own look (a charcoal page with no loading line left, the brand mark, the underlined tab, the app font with its heavier title, the grey edges of the controls), the first view shows the top bar\'s six words, four tiles over the drawing and the check list over the Explain panel, the Design menu holds the details toggle and Load design, a click on a check shows its entry and its lesson link opens its note in the Explain panel, the tab row is Size and torque, Geometry, Plots and Results table and the Plots tab\'s picker shows each plot, the left side on the Geometry tab is still the size controls over the folds Detailed inputs and Assumptions, an input tooltip shows no spreadsheet details and a click on its label shows the input\'s entry in the Explain panel, a click on a dimension tag of the drawing shows its result\'s entry there, a click on the Hot-low tile shows the torque card in the Explain panel, a drag of the gap slider is timed frame by frame (frame_check: true fails a p95 over one 60 fps frame), and the address follows that edit so a reload brings the design back',
  whenToUse: 'Before merging a batch that touched src/gui/ or magcoupling-rs/src/gui/, and during Phase 1 audit',
  phases: [{ title: 'Smoke' }],
}

const SMOKE_SCHEMA = {
  type: 'object', required: ['passed', 'canvas_present', 'console_errors', 'hub_page', 'title_warning'],
  properties: {
    passed: { type: 'boolean' },
    canvas_present: { type: 'boolean' },
    console_errors: { type: 'array', items: { type: 'string' } },
    hub_page: { type: 'boolean' },
    title_warning: { type: 'boolean' },
    screenshot_note: { type: 'string' },
    notes: { type: 'string' },
  },
}

const MAGCOUPLING_SCHEMA = {
  type: 'object', required: ['passed', 'canvas_present', 'console_errors', 'starting_design', 'look', 'share_link_loaded', 'sizing_solved', 'design_file_loaded', 'first_view', 'explorer_ready', 'details_toggle', 'check_entry', 'lesson_link_opened', 'plots_picker', 'same_column', 'tooltip_plain', 'input_entry', 'tag_entry', 'torque_card', 'drag_moved', 'drag_frames_within_60fps', 'address_follows'],
  properties: {
    passed: { type: 'boolean' },
    canvas_present: { type: 'boolean' },
    console_errors: { type: 'array', items: { type: 'string' } },
    starting_design: { type: 'boolean' },
    look: { type: 'boolean' },
    address_follows: { type: 'boolean' },
    first_view: { type: 'boolean' },
    explorer_ready: { type: 'boolean' },
    share_link_loaded: { type: 'boolean' },
    sizing_solved: { type: 'boolean' },
    design_file_loaded: { type: 'boolean' },
    details_toggle: { type: 'boolean' },
    check_entry: { type: 'boolean' },
    lesson_link_opened: { type: 'boolean' },
    plots_picker: { type: 'boolean' },
    same_column: { type: 'boolean' },
    tooltip_plain: { type: 'boolean' },
    input_entry: { type: 'boolean' },
    tag_entry: { type: 'boolean' },
    torque_card: { type: 'boolean' },
    drag_moved: { type: 'boolean' },
    drag_frames: { type: 'integer' },
    drag_frame_ms_median: { type: 'number' },
    drag_frame_ms_p95: { type: 'number' },
    drag_frame_ms_max: { type: 'number' },
    drag_frames_within_60fps: { type: 'boolean' },
    screenshot_note: { type: 'string' },
    notes: { type: 'string' },
  },
}

// What a step's result reads when its agent returns nothing (an agent error): every field its
// schema requires, each check false and each list empty, so a field the schema gains is in the
// fallback too. The drag's numbers are not required, so they stay absent: the drag did not run.
const noReading = schema => ({
  ...Object.fromEntries(schema.required.map(field =>
    [field, schema.properties[field].type === 'array' ? [] : false])),
  notes: 'smoke agent returned no result (agent error)',
})

// The calculator's checks: the booleans its schema requires besides passed and the drag's two,
// which the frame rule judges. This script checks the agent's passed again from them, so a
// reading of torque_card false with passed true fails the run.
const CALCULATOR_CHECKS = MAGCOUPLING_SCHEMA.required.filter(field =>
  MAGCOUPLING_SCHEMA.properties[field].type === 'boolean'
  && !['passed', 'drag_moved', 'drag_frames_within_60fps'].includes(field))

// The design the /tools/magcoupler/ smoke opens through its share link (?m=): the default design
// with the face gap at 1.5 mm, in Torque -> Magnets sizing the axial length to 2.5 N.m.
// magcoupling-rs/src/app.rs (the_smoke_test_share_link_opens_its_design) decodes it and
// compares it with Design::default() plus those settings, so it cannot go stale unnoticed. The
// link carries every input (decision M41-5): regenerate it with MagcouplingPanel::share_link
// when the format changes, a default changes, or an input path is renamed or removed (that
// test fails until then).
const MAGCOUPLING_SMOKE_PAYLOAD = 'lVhLr5w2FP4vrHMRMMO8dm0WrVq1qpRIXVoeMINzARMb7txplP_ec2wDNo9pmkWu5pzP9nk_-BYUQta0Cy5BTW-Z6NuKN7eXnCl-a4IPAW_avlPB5VuQ0YpfJe24aEJatSUlV0laJsnH4PIShVEUJx98UCu6ktWkroPLOTylPhcOfw4ucbjzyRlhTR5cojCe4QsCv4iQ_MYbWiHiPEcwSjohv_YsJn_Cm0kYHbYQiUbE4dEH3GhLclbwhuNPAPhssFDDOlKx5taVWq84mV9hMV3Js9eGKaVhuzBeh915bm86hLuZPjWjqpcstyJbiU8zUB8hNUkPh92RvcwUVi3NwJ1G1HDvMzumQE5Wt-jBBDw4Y4sODG7kVAgAdkXrVoH3K_EwxrGEd66hEnxj35rQVwFGELmmJ6nD0H88Y0YLbvdomfsSgOmN1azpyDvJtY9HXkEzsBQRDSMtZxnDGDnNud1dDFxXyELyzLjcRp4h3yRvSW2V2jn4EqzKNPXkUL8IDoJdRdWRjMusMojksIBsPGeYKpPsDhbfjfRX9iCZaDpQwZo39XhTFO2dpyp-o9pSc480PUigzYHvJyO9lawSNAfp6CjdcXoJuJwVi9sULVj3GC90PaI1IfBDKdeJquJGKNc_qkNbq8LX7g6xRkTv6mBLVHil2SuXQ5aOVCGZG0wD3a8rAzWX_I2RmQamJvkQJwUjnfEDGw6xjuW-EDdGJWFFwTPOmuwxFiuPr9MsuKSemMi5ineiy66GQPaaVw8OjDcNFF60AHGrbDxTD2LjTh9Q0dohwzzZbW7b9HUSsemragV3kzRnRL8N7SLYRIC_NhE1bXp4zCrg9ICnwGXFfQpfLb5PTyzq8AZaq_YjchvgD8tt4P9HbnPiP-VuqexGl_182ieffg22YIPf1mDvpKSyFg3PIGRdzrL_DKymFRWWWTe-RTt2nFQnKIweTEL4QVUC3VkV-jYADEb0hEJBlc59Mib_nJtBFy9Fr-DBVb6qGIOcBoFQXQ9gai9UQiEKEkfkTP74i0KR382E9XHJgDsf13EPzqqc_LQnx8gA93PtVcdA-auinQmtdMmEDpD3UJjfOBSrT6TWt-C_tZuyjg3DGc5mOJ3tlqicNQpvu5Gs3gWXo57S5qBa5H3VK_ILCp5E6dpzdU8khBloix0H57NkXS7VsoxDYSQlA1V_I6-330GPo-6tNYOTIc1pq5NM2BgwldxnFhXFBpxzalJlG2K6YjrnQjTndwq94mZbls9ueSW68fr4tIWoa_94RSaT1mhSNH5yHPhXcKGtODa6U49lknrJAku4pcSxljddOZJgDtixbGi0E8Pi94lP7krJoP0PWu9j_zrLdkawmfZZ37q9Jp2sjhzdyjMhB-Xj6XEYuSumR46F7thfCY7l49xj6ZD8Hfb0toIIs8dGruvdwyRg2V-37FjxghGoCo2ZdaMpsSxb9yuW58M4Dcwk9dmGnAxEO-K3QnFHNYf9jqaGH_bo3hEHmRXM0ybEJ0cmkY8Rb0yiXSfELnUQwwaRS2rGCNPZLRfm2qEYv-ydi6UeXwbTlqJaZJGPGGezgfm15_gqPmAWMeeg4jm2rw6V6-Y2kfC_NiUsLnaTcF_tm2EUdENElbTAXFUgTTYFZoT7jkWYgr_pPstfSd3TDILL8lIAwzPkeKLCPKsjiqglQ7a1ibPJbrpArglxPI1PwagH2_qQ1NFuoL9RKLLuRmHId3a1SQok9DVESwcRAVWsBE-8sVAxzDw7wK4hCGZGmFNePYi6o_k-DvV2G16ALDeYlyFEekkbs4slz06UUExhBTDJacMgtzm9fUox-cahQDwgHtXQKVx4ziALwyvcR8rsi_e5Il1DZoLJzHRYJXqZsYVdDM58HxlW3o_bT6MybwTWaqgFOXn9iegU3T8DV_yV4ehk0afD7hm6hYzGrDDg4zl-BsaRCC6W6EZzID2cNw5kX5LIguL0nKyjYFJj3r54nqH67qFXKHxRcsU2bIWwgvawOJs10PaVBQbDhNZXjlmFM2S6itLFHCZAOdTyVdS9xHyTAj8LeOnoImuuYITJyinwVInFzxuKIOWi48a5Ji8gCZ1x7OXEXp6Dvbvjw7ZMkmWihoKXQ2jh_IBtaCpD7hFdcyqYFUKQBZrxZ52Q8f45qpDgXQ2NTsfDEzA2VnPl3A5zlHPl8bRLtsGmpWrkPj5v42yX1cBnOFuk7YWHLSAOOvARhsEmW5HPSb3HDT2BzSaOto6A_Z3PKZv6l_wGYweEOIdVCBePRVpPWCguNSW7-GDm_HiXrkXxHG-iR584HJ8fwMbgqpkMa8J-Oya8M8aP-uAhPNnVzz1nOxV8eAE1cNtbSDMhaNXDrMB7_FD8DKcVhEFkv40xOxK2HPI3-R212n3_ECj-j14GvwVabt0tr7icBu63D9h4IfWQaj_3dGL8_gnPmYnF-RIEXRXuhglMmY_F3_8F'

// The design file the /tools/magcoupler/ smoke loads through the file picker (rfd's HTML overlay on
// the web): the default design with the face gap at 2 mm. app.rs's smoke test reads it too.
const MAGCOUPLING_SMOKE_DESIGN_FILE = '{"format": "magcoupling-design", "version": 1, "inputs": {"metal.face_gap_mm": 2.0}}'

// The status line of the design the calculator opens on at the clean address (no ?m=), the
// starting design (UX overhaul spec 7.1): typed once here. app.rs's smoke test holds it equal to
// the line the app draws for app::starting_design.
const STARTING_STATUS = 'Starting design: passes all 9 checks'

// The page's background as a browser reports it: the panel colour of the calculator's theme
// (UX overhaul spec 6.1, #181b1f), typed once here. app.rs's smoke test holds it equal to
// app::theme::PANEL.
const PAGE_BACKGROUND = 'rgb(24, 27, 31)'

// The id of the page's loading line (slice B, stage 2, part 2b-iii-b; the stage 1 review's
// S-4), which the app removes when it draws its first frame: typed once here. app.rs's smoke
// test holds it equal to app::LOADING_ID.
const LOADING_ID = 'magcoupling_loading'

// The torque card's heading, which the card step looks for in the Explain panel: typed once here. app.rs's smoke test holds it equal to gui::torque_card::CARD_HEADING.
const CARD_HEADING = 'One step of each size control'

// The Magnet face gap slider's value at the smoke design, as its box shows it before the drag:
// the link's 1.5 mm at the slider's two decimals (its step, 0.01 mm), which egui shows at a
// device pixel ratio of 1 (at 2 it shows 1.5 mm). The drag step finds the slider by it and
// judges drag_moved against it: typed once here. app.rs's smoke test holds it to the link's gap
// and the slider's step.
const SMOKE_GAP_SHOWN = '1.50 mm'

// The link at the top of an input's Explain entry, which the input step finds the entry by (UX
// overhaul spec 5.2): typed once here. app.rs's smoke test holds it equal to
// gui::explain::TRACE_LINK.
const TRACE_LINK = 'Highlight the results this drives'

// The name of the face gap's result, at the top of its Explain entry, which the tag step finds the
// entry by (UX overhaul spec 4.4 and 5.2: a click on a dimension tag selects its result): typed
// once here. app.rs's smoke test holds it equal to gui::plain_text::label_for of the face gap.
const TAG_LABEL = 'Gap at the flat centres (effective gap)'

const PLAYWRIGHT_TOOLS = 'load them via ToolSearch, e.g. "select:mcp__plugin_playwright_playwright__browser_navigate,mcp__plugin_playwright_playwright__browser_snapshot,mcp__plugin_playwright_playwright__browser_console_messages,mcp__plugin_playwright_playwright__browser_take_screenshot,mcp__plugin_playwright_playwright__browser_wait_for,mcp__plugin_playwright_playwright__browser_close"'

// The calculator's step also sets the window to 1280 x 800, clicks the canvas (the Design menu and
// its Load design item, a check's row, a lesson link, the Plots tab and a plot's name in its picker, the Geometry tab, the Detailed inputs fold, an input's label, a dimension tag, a tile), uploads a file, moves the mouse
// (a tooltip, a scroll) and drags the Magnet face gap slider while a timer reads each frame it draws (spec 7.2).
const MAGCOUPLING_TOOLS = 'load them via ToolSearch, e.g. "select:mcp__plugin_playwright_playwright__browser_navigate,mcp__plugin_playwright_playwright__browser_resize,mcp__plugin_playwright_playwright__browser_snapshot,mcp__plugin_playwright_playwright__browser_console_messages,mcp__plugin_playwright_playwright__browser_take_screenshot,mcp__plugin_playwright_playwright__browser_wait_for,mcp__plugin_playwright_playwright__browser_run_code_unsafe,mcp__plugin_playwright_playwright__browser_click,mcp__plugin_playwright_playwright__browser_file_upload,mcp__plugin_playwright_playwright__browser_close"'

// The calculator's drag frame-time check (slice-3 spec 7.2): one 60 fps frame, and the fewest
// drawn drag frames that show the timer caught eframe's drawing (fewer: it missed it).
const FRAME_BUDGET_MS = 16.7
const MIN_DRAG_FRAMES = 30

// The drag's frame timer (slice-3 spec 7.2), installed in the page before the app starts.
// eframe 0.32's web runner asks for an animation frame at every display refresh and draws only
// when something changed (eframe src/web/events.rs, paint_and_schedule): in a drag at 165 Hz,
// 458 of 640 callbacks drew nothing (performance review 4.1). The timer keeps a callback only
// if it made a WebGL draw call (eframe's glow painter draws every frame with drawElements), so
// drag_frames counts drawn frames and the median and the p95 are theirs.
const FRAME_TIMER = `() => { window.__frameMs = []; window.__drew = false; for (const gl of [window.WebGL2RenderingContext, window.WebGLRenderingContext]) { if (!gl) continue; for (const name of ['drawElements', 'drawArrays']) { const draw = gl.prototype[name]; if (typeof draw !== 'function') continue; gl.prototype[name] = function (...args) { window.__drew = true; return draw.apply(this, args); }; } } const raf = window.requestAnimationFrame.bind(window); window.requestAnimationFrame = cb => raf(t => { window.__drew = false; const s = performance.now(); cb(t); const ms = performance.now() - s; if (window.__drew) window.__frameMs.push(ms); }); }`

const ARGS = typeof args === 'string' ? JSON.parse(args || '{}') : (args || {})
if (ARGS.selftest) { return { ok: true } }

const url = ARGS.url || 'http://localhost:8080'
// Only the release run sets frame_check: true (slice-3 spec 7.2 step 4); a drag that did not move
// the gap, drew under MIN_DRAG_FRAMES frames or had a p95 over FRAME_BUDGET_MS then fails the
// run. Otherwise the drag's times are recorded and never fail an ordinary or live run.
// frame_check is true, false or absent: any other value (the string "true", the number 1) stops
// the run here, before its first step, rather than read as false and let a slow drag pass. The
// calculator's result records frame_check, so a run shows whether its check applied.
if (ARGS.frame_check !== undefined && typeof ARGS.frame_check !== 'boolean') {
  const given = JSON.stringify(ARGS.frame_check)
  throw new Error(`gui-smoke: frame_check must be true or false, not ${given}`)
}
const frameCheck = ARGS.frame_check === true

let linkage = null
if (ARGS.linkage !== false) {
  linkage = await agent(
    `Smoke-test the web build at ${url} using Playwright MCP tools (${PLAYWRIGHT_TOOLS}).
Steps: navigate to ${url}/ (the local server forwards it to /tools/, the hub page); snapshot it and collect the console messages at level "info". hub_page=true only if the page is at /tools/ and shows two links, "Linkage Simulator" to /tools/linkage/ and "Magnetic Coupling Calculator" to /tools/magcoupler/.
Then navigate to ${url}/tools/linkage/ (the linkage app); wait 5 seconds for WASM init; snapshot the page and confirm a <canvas> element exists; collect the console messages at level "info" (it includes errors and warnings); take a screenshot and judge whether it shows a rendered app (menu bar / toolbar / panels visible, any theme) vs a blank page. Close the browser.
title_warning=true if any console message contains "Unhandled egui viewport command: Title" (backlog BL-037: there must be none).
passed=true only if: both pages loaded, hub_page, canvas present, title_warning=false, and zero console messages of type error on either page (other warnings are OK — put them in notes). List every console error string verbatim in console_errors. If navigation fails entirely, passed=false with the failure in notes — the server may not be running (caller must have run scripts/serve_web.sh).`,
    { label: 'gui-smoke', phase: 'Smoke', schema: SMOKE_SCHEMA, model: 'sonnet' },
  ) || noReading(SMOKE_SCHEMA)
}

// The magnetic coupling calculator. First (UX overhaul slice A, spec 7.1 and 7.2) the clean
// address, with no share link: it opens the starting design and stays clean, and (slice B,
// stage 1, spec 6) the page wears the calculator's own look: its background is the theme's
// panel colour, read from the page, and the screenshot shows the brand mark before the title,
// the line under the active tab and text in the app's font; since part 2b-iii-b the loading
// line is gone from the page, the title and the left side's heading are heavier than the
// rest, and the controls' edges show in grey. Then
// deterministic state through its share link, checked in
// the console (the page is one canvas). A new page opens on Size and torque (decision G-5): since
// slice B, stage 2 (spec 4.4 and 4.5), the first screenshot shows the four tiles over the drawing
// and the check list over the Explain panel's orientation with no click; since part 2a (spec 4.2)
// the top bar is Undo, Redo, Design, Share, Export and Learn, and the Design menu holds the details
// toggle. Two clicks on the canvas, on the Design menu and its "Load design" item found in
// screenshots, check the web file picker (rfd's HTML overlay) and the file load. Then (spec 5.2
// and 5.3) a click on a check's row shows its entry in the Explain panel, which scrolls on its
// own, and the entry's lesson link shows a teaching note there (decision C-11); the Plots tab
// (part 2b-iii-a, spec 4.4, decision C-9) sits in a tab row of four, and its picker names the five
// plots and shows the one picked; on the Geometry
// tab the left side is still the Size column (part 2b-ii-c, spec 4.3), whose Detailed inputs fold
// the step opens; an input
// row's tooltip shows no spreadsheet details while the toggle is off, and (part 2b-ii, spec 5.2)
// a click on that row's label shows the input's entry in the Explain panel, and (part 2b-ii-b,
// spec 4.4 and 5.2) a click on a dimension tag of the drawing shows its result's entry there. Then (slice 3, spec 7.2 and 8.6) the link is opened again with a timer on every frame the
// app draws: on that fresh page a click on the Hot-low tile shows the torque card's heading,
// caption, Auto line and four rows in the Explain panel (decision C-10), and one drag of the gap
// slider is timed frame by frame. Last (slice A, spec 7.2 and
// 8.1 test 5) the address: the drag was a settled edit, so the app rewrote the address with
// history.replaceState, and a reload of that address brings the edited design back; this is the
// one check of the replaceState call itself. The script checks the agent's
// passed again from its checks (CALCULATOR_CHECKS) and its console errors, judges the drag from
// its numbers (FRAME_BUDGET_MS, MIN_DRAG_FRAMES) and fails the run on the drag only with
// frame_check: true. The Playwright MCP returns only the console messages since the last
// navigation, so each page's messages are collected after its last step, before the next
// navigation or the close. src/app.rs checks that the prompt names the right column's headings,
// the orientation, the entry's headings, the toggle, the note's parts, the tab row and the Plots
// tab's picker, the tab, the row and the
// card's words as the page does, that the card step
// installs FRAME_TIMER, that the drag step reads SMOKE_GAP_SHOWN and the sizing lines' prefix as
// the app has them and that each page's console messages are collected.
// linkage-sim-rs/scripts/gui_smoke_verdict_test.cjs (gate 13) runs this script with a stub agent
// and checks its own rules: the frame timer, frame_check, the calculator's checks, the drag's
// verdict and the fallbacks.
let magcoupling = null
if (ARGS.magcoupling !== false) {
  const page = `${url}/tools/magcoupler/?m=${MAGCOUPLING_SMOKE_PAYLOAD}`
  magcoupling = await agent(
    `Smoke-test the WASM magnetic coupling calculator using Playwright MCP tools (${MAGCOUPLING_TOOLS}).
Steps: first set the window to 1280 x 800 CSS pixels with browser_resize (the laptop size of the first view). Then the starting design: navigate to ${url}/tools/magcoupler/ (the clean address, with no share link) and wait 8 seconds (WASM init). Take a screenshot and read the address's query with browser_run_code_unsafe, code: async (page) => page.evaluate(() => location.search). starting_design=true only if the check list (the right side), under its heading "CHECKS", is topped by the line "${STARTING_STATUS}", the header row at the top of the page (it may wrap to a second line) shows the words "Undo", "Redo", "Design", "Share", "Export" and "Learn" in this order, each without a box around it, and no "Equation panel" or "Open the example design" button, and the code returned an empty string (at the starting design the address stays clean). Then the look: read the page's background with browser_run_code_unsafe, code: async (page) => page.evaluate(() => getComputedStyle(document.body).backgroundColor), and whether the page's loading line is gone, code: async (page) => page.evaluate(() => document.getElementById('${LOADING_ID}') === null). look=true only if the first code returned "${PAGE_BACKGROUND}", the second returned true, and the screenshot shows the calculator's own look: a charcoal page; at the top left a small crimson square right before the title "Magnetic coupling calculator"; in the row of tabs the selected tab, "Size and torque", marked by a crimson line under its name and not by a filled button; the title and the heading "Size and torque" at the top of the left side heavier than the text under them; the unticked box of "Auto: smallest" and the value box at the right end of each slider each showing a grey edge; and every text drawn in one clean sans-serif typeface, with no empty box where a character should be. Collect the console messages at level "info" (this page's: the next step navigates away).
Then navigate to this exact URL (copy it whole; it carries a share link): ${page}
Wait 8 seconds (WASM init, then the sizing solve after its 0.25 s debounce). Snapshot the page and confirm a <canvas> element with id "magcoupling_canvas" exists. Collect the console messages at level "info" (it includes errors and warnings). Take a screenshot and judge whether it shows the calculator rendered in a dark theme (the size controls on the left, the check list and the Explain panel on the right, the Size and torque view in the middle) vs a blank page. first_view=true only if the page shows the first view with no click. The middle region: at its top a row of tabs starting "Size and torque" (the selected tab); under the tabs four tiles side by side, each a large number with its unit over a smaller, quieter label, the second labelled "Hot-low torque" with a line starting "needs" under its label; then a drawing of two rings of coloured magnet blocks inside a grey cup around a shaft, small numbers beside its dimension lines and no list of numbered lines under it. The right side: the heading "CHECKS", a status line, then one row per check, each a small shape, the check's name and a word at the row's right end (OK, Check or Fails); a quiet line under the rows about fixed 3D fields; then the heading "EXPLAIN" and a paragraph starting "Two rings of magnets" with three numbered lines under it. Then the Design menu: click "Design" in the header row (every click as below: browser_run_code_unsafe with page.mouse at X, Y read off a fresh screenshot), wait 1 second and take a screenshot. details_toggle=true only if a menu opened under "Design" listing, top to bottom, "Reset to the starting design", "Open the example design", "Save design", "Load design" and a checkbox labelled "Show spreadsheet details", unticked. Press Escape with code: async (page) => { await page.keyboard.press('Escape'); } and wait 1 second (the menu closes).
Then the design file picker: write this text, exactly, to the file .playwright-mcp/magcoupling-smoke-design.json under the repository root (gitignored; the Playwright MCP uploads only files under the repository) and note its absolute path: ${MAGCOUPLING_SMOKE_DESIGN_FILE}
In a fresh screenshot click "Design" in the header row at the top of the page with browser_run_code_unsafe, code: async (page) => { await page.mouse.click(X, Y); } (X and Y in CSS pixels of the screenshot; the canvas fills the page), wait 1 second, take a screenshot, and click the menu's "Load design" item the same way. rfd shows its overlay (#rfd-overlay: a file input #rfd-input shown as a "Choose File" button, and the buttons "Ok" and "Cancel") and opens the browser's file chooser at once (the tool output reports a "File chooser" modal state); if a snapshot shows the overlay but no chooser opened, click the "Choose File" button. Upload the file with browser_file_upload, click the overlay's "Ok" button, wait 2 seconds and collect the console messages again; delete the file.
design_file_loaded=true only if a console message contains "magcoupling: loaded a design file". If the menu or its item cannot be found or the overlay does not appear after two attempts (each with a fresh screenshot), design_file_loaded=false and say why in notes: the canvas click is the fragile part of this step, and the share-link checks do not depend on it.
Then a check's entry and its lesson link (every click and mouse move below as above: browser_run_code_unsafe with page.mouse at X, Y read off a fresh screenshot): in the check list (the right side) click the name of the row "Hot-low torque". Wait 1 second and take a screenshot. Under the check list the Explain panel now shows the check's entry: a small shape, "Hot-low torque" and its word, then the headings "What it is" and "Numbers behind this check" with the check's numbers under the second. If the line "Full lesson:" at the entry's foot is not in view, move the mouse over the Explain panel and scroll down with code: async (page) => { await page.mouse.move(X, Y); await page.mouse.wheel(0, 1500); await page.waitForTimeout(1000); }, then take a screenshot. check_entry=true only if the entry shows as described, the line "Full lesson:" followed by a link is in view, and the check list's rows above the Explain panel are still in view (the Explain panel scrolls on its own). Then click the link right of "Full lesson:". Wait 1 second and take a screenshot. lesson_link_opened=true only if the Explain panel (the right side, under the check list) now shows a teaching note in place of the check's entry: a bold title, a few sentences, the heading "What to change" with lines that start with Raise, Lower, Choose or Enter, and a closed fold "Where this comes from" (scroll the Explain panel down as above if the fold is not in view).
Then the Plots tab: click "Plots" in the row of tabs at the top of the middle region, wait 1 second and take a screenshot. The row of tabs now reads, left to right on one row, "Size and torque", "Geometry", "Plots" and "Results table", with no plot's name among them; under it a row of the five plots' names, "Torque vs temperature", "Gap sweep", "Pole sweep", "Slip heating" and "Torque vs rotation", the first marked as picked (on a filled background), over a plot. Then click "Slip heating" in that row of names, wait 1 second and take a screenshot. plots_picker=true only if the tab row and the row of names show as described, and after the click "Slip heating" is the name marked as picked and the plot under it has a legend that lists "High case".
Then the left side on another tab: click the "Geometry" tab in the row of tabs at the top of the middle region, wait 1 second and take a screenshot. same_column=true only if the left side shows what it showed on the first view: under the heading "Size and torque" the controls "Required torque", "Minimum OD", "Outside diameter", "Overall length", "Magnet face gap" and "Magnets per ring", each a label over a slider, then two closed folds, "Detailed inputs" and "Assumptions"; and no tab named "Design inputs" at its top. Then an input tooltip: click the fold "Detailed inputs" on the left and wait 1 second. Under it the first group, Key design, lists inputs; in a fresh screenshot move the mouse onto the label of its first row (the row under the heading "Key design": the size control of the same name is above the folds) with code: async (page) => { await page.mouse.move(X, Y); await page.waitForTimeout(1500); }, then take a screenshot. tooltip_plain=true only if a tooltip shows beside that label with at least one sentence about the input and a line that starts "Default:", and no line that is a dotted name such as "metal.face_gap_mm" and no spreadsheet cell such as "Metal design!C119" (the "Show spreadsheet details" toggle is off).
Then click that label (as above, at the X, Y the mouse is at), move the mouse off the label onto the drawing in the middle region, wait 1 second and take a screenshot. input_entry=true only if the Explain panel (the right side, under the check list) now shows the input's entry: at its top the input's value and the label you clicked, under them the link "${TRACE_LINK}", then the heading "What it is" with a sentence about the input.
Then a dimension tag: the drawing in the middle region (the Geometry tab, still selected) has blue dimension lines, each with a small number on a dark plate at its end. In a fresh screenshot click the number "1" on the drawing: the face gap's, a small number on a dark plate at the top of the round view on the left, on the outer ring of magnets just past the outer end of the short blue line across the gap between the rings; not the "1" of the numbered list under the drawing. Move the mouse off it onto an empty part of the middle region, wait 1 second and take a screenshot. tag_entry=true only if the Explain panel (the right side, under the check list) now shows that dimension's entry: at its top a value in mm and the name "${TAG_LABEL}", and further down the link "Equation"; and no Equation panel opened at the foot of the middle region.
If a target of these last checks cannot be found after two attempts (each with a fresh screenshot), set that field false and say why in notes; the check list's last row is "Magnets fit". tag_entry=true was judged last on this page. Collect the console messages at level "info" (the check's entry, the lesson link, the Plots tab, the left side, the tooltip, the input's entry and the tag's entry ran on this page).
Then the torque card and the drag's frame times, on the page opened again. Not on the page above: the design file loaded there has no sizing, so no row reads "Set by Auto" and no Auto line shows; and the tooltip step left the Geometry tab selected, where no tile is drawn. Put a timer on every frame the app draws and open the share link again with browser_run_code_unsafe, code: async (page) => { await page.addInitScript(${FRAME_TIMER}); await page.goto('${page}'); }
Wait 8 seconds (WASM init, then the sizing solve), take a screenshot and click the second tile, "Hot-low torque", in the middle region. Wait 1 second and take a screenshot; if the heading "${CARD_HEADING}" is not in view in the Explain panel (the right side, under the check list), scroll it down as above and take another. torque_card=true only if the Explain panel shows the tile's entry: its number and "Hot-low torque", the heading "What it is" with a sentence about hot-low, the heading "What a change does", then the heading "${CARD_HEADING}" with a link starting "Why:" beside it, and under it, in this order: the line "The change in the model's hot-low torque after one step of one control alone.", the line "Auto sets the magnet length again after a change; each row keeps the length shown.", then four rows: one starting "Outside diameter:", the row "Overall length: Set by Auto; turn Auto off to edit", one starting "Magnet face gap:" and one starting "Magnets per ring:"; and the middle region still shows the four tiles over the drawing. A row may wrap onto more lines.
Then the drag: in a fresh screenshot find the handle of the "Magnet face gap" slider among the size controls on the left (it reads ${SMOKE_GAP_SHOWN}; Auto on the magnet length leaves this slider free). Drag it and read the frame times in one call of browser_run_code_unsafe, code: async (page) => { const x = X, y = Y; let sizing = 0; const onConsole = m => { if (m.text().includes('magcoupling sizing: ')) sizing += 1; }; page.on('console', onConsole); await page.mouse.move(x, y); await page.evaluate(() => { window.__frameMs = []; }); await page.mouse.down(); for (let i = 0; i < 60; i++) { await page.mouse.move(i % 2 === 0 ? x + 3 : x - 3, y); await page.waitForTimeout(50); } await page.mouse.up(); const ms = await page.evaluate(() => window.__frameMs.slice()); page.off('console', onConsole); const s = [...ms].sort((a, b) => a - b); const rank = q => (s.length ? s[Math.ceil(q * s.length) - 1] : null); return { frames: s.length, median: rank(0.5), p95: rank(0.95), max: s.length ? s[s.length - 1] : null, sizing_lines: sizing }; } (X and Y the handle's centre in CSS pixels of the screenshot; the timer is emptied just before the press, so only the drag's frames count, not the load, the solve or the screenshots). Then take a screenshot. drag_moved=true only if the Magnet face gap now reads a value other than ${SMOKE_GAP_SHOWN} (the drag held the handle). From the returned object: drag_frames=frames, drag_frame_ms_median=median, drag_frame_ms_p95=p95 and drag_frame_ms_max=max (milliseconds per frame, nearest rank). drag_frames_within_60fps=true only if drag_moved, drag_frames >= ${MIN_DRAG_FRAMES} and drag_frame_ms_p95 <= ${FRAME_BUDGET_MS} (this script checks it again from your numbers); under ${MIN_DRAG_FRAMES} frames the timer saw the app draw too few frames: say so in notes. In notes give sizing_lines, the number of "magcoupling sizing: " console lines logged during the drag (expected 0: the moves come 50 ms apart, so the 0.25 s debounce never runs a solve mid-drag). If the handle cannot be found after two attempts (each with a fresh screenshot), set drag_moved=false, leave the four numbers out and say why in notes.
Collect the console messages at level "info" once more (it covers the reopened page and the drag).
Then the address. The app rewrites it after every settled edit, and the drag was one. Read it, reload the page and read it again in one call of browser_run_code_unsafe, code: async (page) => { const before = await page.evaluate(() => location.href); await page.reload(); await page.waitForTimeout(8000); return { before, after: await page.evaluate(() => location.href) }; } Then take a screenshot. address_follows=true only if drag_moved, before contains "?m=" and is not the URL opened above (the drag's edit rewrote the address), after equals before, and on the reloaded page the Magnet face gap reads the value it read after the drag, not ${SMOKE_GAP_SHOWN} (the reload brought the edited design back). If drag_moved is false, address_follows=false: say so in notes. Collect the console messages at level "info" a last time (the reloaded page), then close the browser.
share_link_loaded=true only if a console message contains "magcoupling: loaded the design from the share link". sizing_solved=true only if a console message contains "magcoupling sizing: Solved at". explorer_ready=true only if a console message contains "magcoupling explorer: " (the equation registry, built when the page starts).
passed=true only if: page loaded, canvas present, starting_design, look, first_view, details_toggle, share_link_loaded, sizing_solved, design_file_loaded, explorer_ready, check_entry, lesson_link_opened, plots_picker, same_column, tooltip_plain, input_entry, tag_entry, torque_card, address_follows, and zero console messages of type error (warnings are OK — put them in notes); leave the drag's fields out of passed (this script judges them). List every console error string verbatim in console_errors. If navigation fails entirely, passed=false with the failure in notes — the server may not be running (serve linkage-sim-rs/web/ after running scripts/build_magcoupling_web.sh).`,
    { label: 'gui-smoke-magcoupling', phase: 'Smoke', schema: MAGCOUPLING_SCHEMA, model: 'sonnet' },
  ) || noReading(MAGCOUPLING_SCHEMA)
  // The calculator's passed, checked again by this script from the agent's reading rather than
  // taken on its word: the agent's own passed (it alone sees that the page loaded), every check
  // true and no console error listed.
  magcoupling.passed = magcoupling.passed === true
    && CALCULATOR_CHECKS.every(field => magcoupling[field] === true)
    && Array.isArray(magcoupling.console_errors) && magcoupling.console_errors.length === 0
  // The drag's verdict (slice-3 spec 7.2 steps 3 and 4), from its numbers by this script, not the
  // agent's arithmetic: recorded on every run, and failing the run only with frame_check: true.
  const p95 = magcoupling.drag_frame_ms_p95
  magcoupling.drag_frames_within_60fps = magcoupling.drag_moved === true
    && Number.isInteger(magcoupling.drag_frames) && magcoupling.drag_frames >= MIN_DRAG_FRAMES
    && Number.isFinite(p95) && p95 <= FRAME_BUDGET_MS
  magcoupling.frame_check = frameCheck
  if (frameCheck) magcoupling.passed = magcoupling.passed && magcoupling.drag_frames_within_60fps
}

// The linkage app's fields at the top level, as before, with the calculator's beside them.
const passed = (linkage ? linkage.passed : true) && (magcoupling ? magcoupling.passed : true)
return { ...(linkage || {}), passed, linkage, magcoupling }
