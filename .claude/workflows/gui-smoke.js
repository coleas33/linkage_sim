export const meta = {
  name: 'gui-smoke',
  description: 'Smoke-test the locally served web build (the hub page at /, the linkage app at /linkage/, the calculator at /magcoupler/): loads, canvas present, console clean',
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
  type: 'object', required: ['passed', 'canvas_present', 'console_errors', 'share_link_loaded', 'sizing_solved', 'design_file_loaded', 'geometry_view', 'explorer_ready'],
  properties: {
    passed: { type: 'boolean' },
    canvas_present: { type: 'boolean' },
    console_errors: { type: 'array', items: { type: 'string' } },
    geometry_view: { type: 'boolean' },
    explorer_ready: { type: 'boolean' },
    share_link_loaded: { type: 'boolean' },
    sizing_solved: { type: 'boolean' },
    design_file_loaded: { type: 'boolean' },
    screenshot_note: { type: 'string' },
    notes: { type: 'string' },
  },
}

// The design the /magcoupler/ smoke opens through its share link (?m=): the default design
// with the face gap at 1.5 mm, in Torque -> Magnets sizing the axial length to 2.5 N.m.
// magcoupling-rs/src/app.rs (the_smoke_test_share_link_opens_its_design) decodes it and
// compares it with Design::default() plus those settings, so it cannot go stale unnoticed. The
// link carries every input (decision M41-5): regenerate it with MagcouplingPanel::share_link
// when the format changes, a default changes, or an input path is renamed or removed (that
// test fails until then).
const MAGCOUPLING_SMOKE_PAYLOAD = 'lVhLr5w2FP4vrHMRMMO8dm0WrVq1qpRIXVoeMINzARMb7txplP_ec2wDNo9pmkWu5pzP9nk_-BYUQta0Cy5BTW-Z6NuKN7eXnCl-a4IPAW_avlPB5VuQ0YpfJe24aEJatSUlV0laJsnH4PIShVEUJx98UCu6ktWkroPLOTylPhcOfw4ucbjzyRlhTR5cojCe4QsCv4iQ_MYbWiHiPEcwSjohv_YsJn_Cm0kYHbYQiUbE4dEH3GhLclbwhuNPAPhssFDDOlKx5taVWq84mV9hMV3Js9eGKaVhuzBeh915bm86hLuZPjWjqpcstyJbiU8zUB8hNUkPh92RvcwUVi3NwJ1G1HDvMzumQE5Wt-jBBDw4Y4sODG7kVAgAdkXrVoH3K_EwxrGEd66hEnxj35rQVwFGELmmJ6nD0H88Y0YLbvdomfsSgOmN1azpyDvJtY9HXkEzsBQRDSMtZxnDGDnNud1dDFxXyELyzLjcRp4h3yRvSW2V2jn4EqzKNPXkUL8IDoJdRdWRjMusMojksIBsPGeYKpPsDhbfjfRX9iCZaDpQwZo39XhTFO2dpyp-o9pSc480PUigzYHvJyO9lawSNAfp6CjdcXoJuJwVi9sULVj3GC90PaI1IfBDKdeJquJGKNc_qkNbq8LX7g6xRkTv6mBLVHil2SuXQ5aOVCGZG0wD3a8rAzWX_I2RmQamJvkQJwUjnfEDGw6xjuW-EDdGJWFFwTPOmuwxFiuPr9MsuKSemMi5ineiy66GQPaaVw8OjDcNFF60AHGrbDxTD2LjTh9Q0dohwzzZbW7b9HUSsemragV3kzRnRL8N7SLYRIC_NhE1bXp4zCrg9ICnwGXFfQpfLb5PTyzq8AZaq_YjchvgD8tt4P9HbnPiP-VuqexGl_182ieffg22YIPf1mDvpKSyFg3PIGRdzrL_DKymFRWWWTe-RTt2nFQnKIweTEL4QVUC3VkV-jYADEb0hEJBlc59Mib_nJtBFy9Fr-DBVb6qGIOcBoFQXQ9gai9UQiEKEkfkTP74i0KR382E9XHJgDsf13EPzqqc_LQnx8gA93PtVcdA-auinQmtdMmEDpD3UJjfOBSrT6TWt-C_tZuyjg3DGc5mOJ3tlqicNQpvu5Gs3gWXo57S5qBa5H3VK_ILCp5E6dpzdU8khBloix0H57NkXS7VsoxDYSQlA1V_I6-330GPo-6tNYOTIc1pq5NM2BgwldxnFhXFBpxzalJlG2K6YjrnQjTndwq94mZbls9ueSW68fr4tIWoa_94RSaT1mhSNH5yHPhXcKGtODa6U49lknrJAku4pcSxljddOZJgDtixbGi0E8Pi94lP7krJoP0PWu9j_zrLdkawmfZZ37q9Jp2sjhzdyjMhB-Xj6XEYuSumR46F7thfCY7l49xj6ZD8Hfb0toIIs8dGruvdwyRg2V-37FjxghGoCo2ZdaMpsSxb9yuW58M4Dcwk9dmGnAxEO-K3QnFHNYf9jqaGH_bo3hEHmRXM0ybEJ0cmkY8Rb0yiXSfELnUQwwaRS2rGCNPZLRfm2qEYv-ydi6UeXwbTlqJaZJGPGGezgfm15_gqPmAWMeeg4jm2rw6V6-Y2kfC_NiUsLnaTcF_tm2EUdENElbTAXFUgTTYFZoT7jkWYgr_pPstfSd3TDILL8lIAwzPkeKLCPKsjiqglQ7a1ibPJbrpArglxPI1PwagH2_qQ1NFuoL9RKLLuRmHId3a1SQok9DVESwcRAVWsBE-8sVAxzDw7wK4hCGZGmFNePYi6o_k-DvV2G16ALDeYlyFEekkbs4slz06UUExhBTDJacMgtzm9fUox-cahQDwgHtXQKVx4ziALwyvcR8rsi_e5Il1DZoLJzHRYJXqZsYVdDM58HxlW3o_bT6MybwTWaqgFOXn9iegU3T8DV_yV4ehk0afD7hm6hYzGrDDg4zl-BsaRCC6W6EZzID2cNw5kX5LIguL0nKyjYFJj3r54nqH67qFXKHxRcsU2bIWwgvawOJs10PaVBQbDhNZXjlmFM2S6itLFHCZAOdTyVdS9xHyTAj8LeOnoImuuYITJyinwVInFzxuKIOWi48a5Ji8gCZ1x7OXEXp6Dvbvjw7ZMkmWihoKXQ2jh_IBtaCpD7hFdcyqYFUKQBZrxZ52Q8f45qpDgXQ2NTsfDEzA2VnPl3A5zlHPl8bRLtsGmpWrkPj5v42yX1cBnOFuk7YWHLSAOOvARhsEmW5HPSb3HDT2BzSaOto6A_Z3PKZv6l_wGYweEOIdVCBePRVpPWCguNSW7-GDm_HiXrkXxHG-iR584HJ8fwMbgqpkMa8J-Oya8M8aP-uAhPNnVzz1nOxV8eAE1cNtbSDMhaNXDrMB7_FD8DKcVhEFkv40xOxK2HPI3-R212n3_ECj-j14GvwVabt0tr7icBu63D9h4IfWQaj_3dGL8_gnPmYnF-RIEXRXuhglMmY_F3_8F'

// The design file the /magcoupler/ smoke loads through the file picker (rfd's HTML overlay on
// the web): the default design with the face gap at 2 mm. app.rs's smoke test reads it too.
const MAGCOUPLING_SMOKE_DESIGN_FILE = '{"format": "magcoupling-design", "version": 1, "inputs": {"metal.face_gap_mm": 2.0}}'

const PLAYWRIGHT_TOOLS = 'load them via ToolSearch, e.g. "select:mcp__plugin_playwright_playwright__browser_navigate,mcp__plugin_playwright_playwright__browser_snapshot,mcp__plugin_playwright_playwright__browser_console_messages,mcp__plugin_playwright_playwright__browser_take_screenshot,mcp__plugin_playwright_playwright__browser_wait_for,mcp__plugin_playwright_playwright__browser_close"'

// The calculator's step also clicks the canvas once and uploads a file.
const MAGCOUPLING_TOOLS = 'load them via ToolSearch, e.g. "select:mcp__plugin_playwright_playwright__browser_navigate,mcp__plugin_playwright_playwright__browser_snapshot,mcp__plugin_playwright_playwright__browser_console_messages,mcp__plugin_playwright_playwright__browser_take_screenshot,mcp__plugin_playwright_playwright__browser_wait_for,mcp__plugin_playwright_playwright__browser_run_code_unsafe,mcp__plugin_playwright_playwright__browser_click,mcp__plugin_playwright_playwright__browser_file_upload,mcp__plugin_playwright_playwright__browser_close"'

const ARGS = typeof args === 'string' ? JSON.parse(args || '{}') : (args || {})
if (ARGS.selftest) { return { ok: true } }

const url = ARGS.url || 'http://localhost:8080'

let linkage = null
if (ARGS.linkage !== false) {
  linkage = await agent(
    `Smoke-test the web build at ${url} using Playwright MCP tools (${PLAYWRIGHT_TOOLS}).
Steps: navigate to ${url}/ (the hub page); snapshot it. hub_page=true only if it shows two links, "Linkage Simulator" to /linkage/ and "Magnetic Coupling Calculator" to /magcoupler/.
Then navigate to ${url}/linkage/ (the linkage app); wait 5 seconds for WASM init; snapshot the page and confirm a <canvas> element exists; collect the console messages at level "info" (it includes errors and warnings); take a screenshot and judge whether it shows a rendered app (menu bar / toolbar / panels visible, any theme) vs a blank page. Close the browser.
title_warning=true if any console message contains "Unhandled egui viewport command: Title" (backlog BL-037: there must be none).
passed=true only if: both pages loaded, hub_page, canvas present, title_warning=false, and zero console messages of type error on either page (other warnings are OK — put them in notes). List every console error string verbatim in console_errors. If navigation fails entirely, passed=false with the failure in notes — the server may not be running (caller must have run scripts/serve_web.sh).`,
    { label: 'gui-smoke', phase: 'Smoke', schema: SMOKE_SCHEMA, model: 'sonnet' },
  ) || { passed: false, canvas_present: false, console_errors: [], hub_page: false, title_warning: false, notes: 'smoke agent returned no result (agent error)' }
}

// The magnetic coupling calculator: deterministic state through its share link, checked in
// the console (the page is one canvas). The geometry view is the centre region's default view,
// so the first screenshot shows it with no click. One click on the canvas, on the "Load design"
// button found in a screenshot, checks the web file picker (rfd's HTML overlay) and the file load.
let magcoupling = null
if (ARGS.magcoupling !== false) {
  const page = `${url}/magcoupler/?m=${MAGCOUPLING_SMOKE_PAYLOAD}`
  magcoupling = await agent(
    `Smoke-test the WASM magnetic coupling calculator using Playwright MCP tools (${MAGCOUPLING_TOOLS}).
Steps: navigate to this exact URL (copy it whole; it carries a share link): ${page}
Wait 8 seconds (WASM init, then the sizing solve after its 0.25 s debounce). Snapshot the page and confirm a <canvas> element with id "magcoupling_canvas" exists. Collect the console messages at level "info" (it includes errors and warnings). Take a screenshot and judge whether it shows the calculator rendered in a dark theme (inputs on the left, dashboard on the right, the geometry view in the middle) vs a blank page. geometry_view=true only if the middle region shows the geometry view: a row of tabs starting "Geometry", under it two drawings side by side (an end view of two rings of coloured magnet blocks inside a grey cup around a shaft, and a half side view with dashed space-claim lines), and a numbered list of dimension callouts under them ("1 Face gap ...", "4 Overall length: ...").
Then the design file picker, the only click on the canvas: write this text, exactly, to the file .playwright-mcp/magcoupling-smoke-design.json under the repository root (gitignored; the Playwright MCP uploads only files under the repository) and note its absolute path: ${MAGCOUPLING_SMOKE_DESIGN_FILE}
In the screenshot, find the "Load design" button in the header row at the top of the page (between "Save design" and "Copy share link") and click its centre with browser_run_code_unsafe, code: async (page) => { await page.mouse.click(X, Y); } (X and Y in CSS pixels of the screenshot; the canvas fills the page). rfd shows its overlay (#rfd-overlay: a file input #rfd-input shown as a "Choose File" button, and the buttons "Ok" and "Cancel") and opens the browser's file chooser at once (the tool output reports a "File chooser" modal state); if a snapshot shows the overlay but no chooser opened, click the "Choose File" button. Upload the file with browser_file_upload, click the overlay's "Ok" button, wait 2 seconds and collect the console messages again; delete the file.
design_file_loaded=true only if a console message contains "magcoupling: loaded a design file". If the button cannot be found or the overlay does not appear after two attempts (each with a fresh screenshot), design_file_loaded=false and say why in notes: the canvas click is the fragile part of this step, and the share-link checks do not depend on it. Close the browser.
share_link_loaded=true only if a console message contains "magcoupling: loaded the design from the share link". sizing_solved=true only if a console message contains "magcoupling sizing: Solved at". explorer_ready=true only if a console message contains "magcoupling explorer: " (the equation registry, built when the page starts).
passed=true only if: page loaded, canvas present, geometry_view, share_link_loaded, sizing_solved, design_file_loaded, explorer_ready, and zero console messages of type error (warnings are OK — put them in notes). List every console error string verbatim in console_errors. If navigation fails entirely, passed=false with the failure in notes — the server may not be running (serve linkage-sim-rs/web/ after running scripts/build_magcoupling_web.sh).`,
    { label: 'gui-smoke-magcoupling', phase: 'Smoke', schema: MAGCOUPLING_SCHEMA, model: 'sonnet' },
  ) || { passed: false, canvas_present: false, console_errors: [], geometry_view: false, explorer_ready: false, share_link_loaded: false, sizing_solved: false, design_file_loaded: false, notes: 'smoke agent returned no result (agent error)' }
}

// The linkage app's fields at the top level, as before, with the calculator's beside them.
const passed = (linkage ? linkage.passed : true) && (magcoupling ? magcoupling.passed : true)
return { ...(linkage || {}), passed, linkage, magcoupling }
