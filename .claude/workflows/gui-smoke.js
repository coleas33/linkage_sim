export const meta = {
  name: 'gui-smoke',
  description: 'Smoke-test the locally served web build (the hub page at /tools/, the linkage app at /tools/linkage/, the calculator at /tools/magcoupler/): loads, canvas present, console clean; on the calculator a Why link opens a note, an input tooltip shows no spreadsheet details, the dashboard scrolls at 1280x800, the torque card shows its rows between the readouts and the drawing, and a drag of the gap slider is timed frame by frame (frame_check: true fails a p95 over one 60 fps frame)',
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
  type: 'object', required: ['passed', 'canvas_present', 'console_errors', 'share_link_loaded', 'sizing_solved', 'design_file_loaded', 'first_view', 'explorer_ready', 'details_toggle', 'why_link_opened', 'tooltip_plain', 'dashboard_scrolls', 'torque_card', 'drag_moved', 'drag_frames_within_60fps'],
  properties: {
    passed: { type: 'boolean' },
    canvas_present: { type: 'boolean' },
    console_errors: { type: 'array', items: { type: 'string' } },
    first_view: { type: 'boolean' },
    explorer_ready: { type: 'boolean' },
    share_link_loaded: { type: 'boolean' },
    sizing_solved: { type: 'boolean' },
    design_file_loaded: { type: 'boolean' },
    details_toggle: { type: 'boolean' },
    why_link_opened: { type: 'boolean' },
    tooltip_plain: { type: 'boolean' },
    dashboard_scrolls: { type: 'boolean' },
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

// The torque card's heading, which the calculator's first view and its card step both look for:
// typed once here. app.rs's smoke test holds it equal to gui::torque_card::CARD_HEADING.
const CARD_HEADING = 'One step of each size control'

const PLAYWRIGHT_TOOLS = 'load them via ToolSearch, e.g. "select:mcp__plugin_playwright_playwright__browser_navigate,mcp__plugin_playwright_playwright__browser_snapshot,mcp__plugin_playwright_playwright__browser_console_messages,mcp__plugin_playwright_playwright__browser_take_screenshot,mcp__plugin_playwright_playwright__browser_wait_for,mcp__plugin_playwright_playwright__browser_close"'

// The calculator's step also sets the window to 1280 x 800, clicks the canvas (the Load design
// button, a Why link, the Geometry tab), uploads a file, moves the mouse (a tooltip, a scroll)
// and drags the Magnet face gap slider while a timer reads each frame it draws (spec 7.2).
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
  ) || { passed: false, canvas_present: false, console_errors: [], hub_page: false, title_warning: false, notes: 'smoke agent returned no result (agent error)' }
}

// The magnetic coupling calculator: deterministic state through its share link, checked in
// the console (the page is one canvas). A new page opens on Size and torque under the welcome
// strip (decision G-5), so the first screenshot shows both with no click. One click on the
// canvas, on the "Load design" button found in a screenshot, checks the web file picker (rfd's
// HTML overlay) and the file load. Then (spec R14, guidance slice 2) a dashboard Why link opens
// a teaching note, an input row's tooltip shows no spreadsheet details while the toggle is
// off, and the dashboard scrolls to its last row at 1280 x 800. Then (slice 3, spec 7.2 and
// 8.6) the link is opened again with a timer on every frame the app draws: on that fresh page the
// torque card shows its heading, caption, Auto line and four rows between the readouts and the
// drawing, and one drag of the gap slider is timed frame by frame. The script judges the drag
// from its numbers (FRAME_BUDGET_MS, MIN_DRAG_FRAMES) and fails the run on it only with
// frame_check: true. The Playwright MCP returns only the console messages since the last
// navigation, so each page's messages are collected after its last step, before the next
// navigation or the close. src/app.rs checks that the prompt names the strip, the toggle, the
// note's parts, the tab, the row and the card's words as the page does, that the card step
// installs FRAME_TIMER and that each page's console messages are collected.
let magcoupling = null
if (ARGS.magcoupling !== false) {
  const page = `${url}/tools/magcoupler/?m=${MAGCOUPLING_SMOKE_PAYLOAD}`
  magcoupling = await agent(
    `Smoke-test the WASM magnetic coupling calculator using Playwright MCP tools (${MAGCOUPLING_TOOLS}).
Steps: first set the window to 1280 x 800 CSS pixels with browser_resize (the laptop size the dashboard must scroll at). Then navigate to this exact URL (copy it whole; it carries a share link): ${page}
Wait 8 seconds (WASM init, then the sizing solve after its 0.25 s debounce). Snapshot the page and confirm a <canvas> element with id "magcoupling_canvas" exists. Collect the console messages at level "info" (it includes errors and warnings). Take a screenshot and judge whether it shows the calculator rendered in a dark theme (the size controls on the left, dashboard on the right, the Size and torque view in the middle) vs a blank page. first_view=true only if the middle region shows the first view with no click: at its top the line "Set the size -> read the torque -> open any red check to see why and what to change." with a "Hide" button, under it a row of tabs starting "Size and torque" (the selected tab), and under the tabs the torque readouts, then the heading "${CARD_HEADING}" with a link starting "Why:" beside it, open, with lines of text under it, then a drawing of two rings of coloured magnet blocks inside a grey cup around a shaft. details_toggle=true only if the header row at the top of the page (it may wrap to a second line) shows, after the "Equation panel" button, a checkbox labelled "Show spreadsheet details", unticked.
Then the design file picker: write this text, exactly, to the file .playwright-mcp/magcoupling-smoke-design.json under the repository root (gitignored; the Playwright MCP uploads only files under the repository) and note its absolute path: ${MAGCOUPLING_SMOKE_DESIGN_FILE}
In the screenshot, find the "Load design" button in the header row at the top of the page (between "Save design" and "Copy share link") and click its centre with browser_run_code_unsafe, code: async (page) => { await page.mouse.click(X, Y); } (X and Y in CSS pixels of the screenshot; the canvas fills the page). rfd shows its overlay (#rfd-overlay: a file input #rfd-input shown as a "Choose File" button, and the buttons "Ok" and "Cancel") and opens the browser's file chooser at once (the tool output reports a "File chooser" modal state); if a snapshot shows the overlay but no chooser opened, click the "Choose File" button. Upload the file with browser_file_upload, click the overlay's "Ok" button, wait 2 seconds and collect the console messages again; delete the file.
design_file_loaded=true only if a console message contains "magcoupling: loaded a design file". If the button cannot be found or the overlay does not appear after two attempts (each with a fresh screenshot), design_file_loaded=false and say why in notes: the canvas click is the fragile part of this step, and the share-link checks do not depend on it.
Then a teaching note through a Why link (every click and mouse move below as above: browser_run_code_unsafe with page.mouse at X, Y read off a fresh screenshot): on the dashboard (the right side), under its first row (the nominal pull-out torque), find the line that starts "Why:" and click the link text right of "Why:". Wait 1 second and take a screenshot. why_link_opened=true only if the Equation panel is now open, docked at the bottom of the middle region, and shows a teaching note: a bold title, a few sentences, the heading "What to change" with lines that start with Raise, Lower, Choose or Enter, and a closed fold "Where this comes from".
Then an input tooltip: click the "Geometry" tab in the row of tabs at the top of the middle region and wait 1 second. On the left the first group, Key design, lists inputs; in a fresh screenshot move the mouse onto the label of its first row with code: async (page) => { await page.mouse.move(X, Y); await page.waitForTimeout(1500); }, then take a screenshot. tooltip_plain=true only if a tooltip shows beside that label with at least one sentence about the input and a line that starts "Default:", and no line that is a dotted name such as "metal.face_gap_mm" and no spreadsheet cell such as "Metal design!C119" (the "Show spreadsheet details" toggle is off).
Then the dashboard's scrolling: in a fresh screenshot move the mouse to the middle of the dashboard and scroll down with code: async (page) => { await page.mouse.move(X, Y); await page.mouse.wheel(0, 4000); await page.waitForTimeout(1000); }, then take a screenshot. dashboard_scrolls=true only if the dashboard's last row, "Magnets fit", and the "Why:" line under it are in view and its first row has scrolled out of view.
If a target of these last three checks cannot be found after two attempts (each with a fresh screenshot), set that field false and say why in notes. Collect the console messages at level "info" (the Why link, the tooltip and the scroll ran on this page).
Then the torque card and the drag's frame times, on the page opened again. Not on the page above: the design file loaded there has no sizing, so no row reads "Set by Auto" and no Auto line shows; the dashboard's Why link opened the Equation panel, which closes the card; and the tooltip step left the Geometry tab selected, where no card is drawn. Put a timer on every frame the app draws and open the share link again with browser_run_code_unsafe, code: async (page) => { await page.addInitScript(${FRAME_TIMER}); await page.goto('${page}'); }
Wait 8 seconds (WASM init, then the sizing solve) and take a screenshot. torque_card=true only if the middle region shows, between the torque readouts and the drawing, the heading "${CARD_HEADING}" with a link starting "Why:" beside it, open, and under it, in this order: the line "The change in the model's hot-low torque after one step of one control alone.", the line "Auto sets the magnet length again after a change; each row keeps the length shown.", then four rows: one starting "Outside diameter:", the row "Overall length: Set by Auto; turn Auto off to edit", one starting "Magnet face gap:" and one starting "Magnets per ring:"; and the drawing still shows below the card. A row may wrap onto more lines.
Then the drag: in a fresh screenshot find the handle of the "Magnet face gap" slider among the size controls on the left (it reads 1.50 mm; Auto on the magnet length leaves this slider free). Drag it and read the frame times in one call of browser_run_code_unsafe, code: async (page) => { const x = X, y = Y; let sizing = 0; const onConsole = m => { if (m.text().includes('magcoupling sizing: ')) sizing += 1; }; page.on('console', onConsole); await page.mouse.move(x, y); await page.evaluate(() => { window.__frameMs = []; }); await page.mouse.down(); for (let i = 0; i < 60; i++) { await page.mouse.move(i % 2 === 0 ? x + 3 : x - 3, y); await page.waitForTimeout(50); } await page.mouse.up(); const ms = await page.evaluate(() => window.__frameMs.slice()); page.off('console', onConsole); const s = [...ms].sort((a, b) => a - b); const rank = q => (s.length ? s[Math.ceil(q * s.length) - 1] : null); return { frames: s.length, median: rank(0.5), p95: rank(0.95), max: s.length ? s[s.length - 1] : null, sizing_lines: sizing }; } (X and Y the handle's centre in CSS pixels of the screenshot; the timer is emptied just before the press, so only the drag's frames count, not the load, the solve or the screenshots). Then take a screenshot. drag_moved=true only if the Magnet face gap now reads a value other than 1.50 mm (the drag held the handle). From the returned object: drag_frames=frames, drag_frame_ms_median=median, drag_frame_ms_p95=p95 and drag_frame_ms_max=max (milliseconds per frame, nearest rank). drag_frames_within_60fps=true only if drag_moved, drag_frames >= ${MIN_DRAG_FRAMES} and drag_frame_ms_p95 <= ${FRAME_BUDGET_MS} (this script checks it again from your numbers); under ${MIN_DRAG_FRAMES} frames the timer saw the app draw too few frames: say so in notes. In notes give sizing_lines, the number of "magcoupling sizing: " console lines logged during the drag (expected 0: the moves come 50 ms apart, so the 0.25 s debounce never runs a solve mid-drag). If the handle cannot be found after two attempts (each with a fresh screenshot), set drag_moved=false, leave the four numbers out and say why in notes.
Collect the console messages at level "info" once more (it covers the reopened page and the drag), then close the browser.
share_link_loaded=true only if a console message contains "magcoupling: loaded the design from the share link". sizing_solved=true only if a console message contains "magcoupling sizing: Solved at". explorer_ready=true only if a console message contains "magcoupling explorer: " (the equation registry, built when the page starts).
passed=true only if: page loaded, canvas present, first_view, details_toggle, share_link_loaded, sizing_solved, design_file_loaded, explorer_ready, why_link_opened, tooltip_plain, dashboard_scrolls, torque_card, and zero console messages of type error (warnings are OK — put them in notes); leave the drag's fields out of passed (this script judges them). List every console error string verbatim in console_errors. If navigation fails entirely, passed=false with the failure in notes — the server may not be running (serve linkage-sim-rs/web/ after running scripts/build_magcoupling_web.sh).`,
    { label: 'gui-smoke-magcoupling', phase: 'Smoke', schema: MAGCOUPLING_SCHEMA, model: 'sonnet' },
  ) || { passed: false, canvas_present: false, console_errors: [], first_view: false, explorer_ready: false, share_link_loaded: false, sizing_solved: false, design_file_loaded: false, details_toggle: false, why_link_opened: false, tooltip_plain: false, dashboard_scrolls: false, torque_card: false, drag_moved: false, drag_frames: 0, drag_frames_within_60fps: false, notes: 'smoke agent returned no result (agent error)' }
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
