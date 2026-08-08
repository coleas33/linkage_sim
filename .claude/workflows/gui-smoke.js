export const meta = {
  name: 'gui-smoke',
  description: 'Smoke-test the locally served WASM build: loads, canvas present, console clean',
  whenToUse: 'Before merging a batch that touched src/gui/, and during Phase 1 audit',
  phases: [{ title: 'Smoke' }],
}

const SMOKE_SCHEMA = {
  type: 'object', required: ['passed', 'canvas_present', 'console_errors'],
  properties: {
    passed: { type: 'boolean' },
    canvas_present: { type: 'boolean' },
    console_errors: { type: 'array', items: { type: 'string' } },
    screenshot_note: { type: 'string' },
    notes: { type: 'string' },
  },
}

const ARGS = typeof args === 'string' ? JSON.parse(args || '{}') : (args || {})
if (ARGS.selftest) { return { ok: true } }

const url = ARGS.url || 'http://localhost:8080'
const result = await agent(
  `Smoke-test the WASM linkage app at ${url} using Playwright MCP tools (load them via ToolSearch, e.g. "select:mcp__plugin_playwright_playwright__browser_navigate,mcp__plugin_playwright_playwright__browser_snapshot,mcp__plugin_playwright_playwright__browser_console_messages,mcp__plugin_playwright_playwright__browser_take_screenshot,mcp__plugin_playwright_playwright__browser_wait_for,mcp__plugin_playwright_playwright__browser_close").
Steps: navigate to ${url}; wait 5 seconds for WASM init; snapshot the page and confirm a <canvas> element exists; collect console messages; take a screenshot and judge whether it shows a rendered app (menu bar / toolbar / panels visible, any theme) vs a blank page; close the browser.
passed=true only if: page loaded, canvas present, zero console messages of type error (warnings are OK — put them in notes). List every console error string verbatim in console_errors. If navigation fails entirely, passed=false with the failure in notes — the server may not be running (caller must have run scripts/serve_web.sh).`,
  { label: 'gui-smoke', phase: 'Smoke', schema: SMOKE_SCHEMA },
)
return result || { passed: false, canvas_present: false, console_errors: [], notes: 'smoke agent returned no result (agent error)' }
