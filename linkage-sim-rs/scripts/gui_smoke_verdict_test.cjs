// The gui-smoke workflow's own rules (.claude/workflows/gui-smoke.js), checked under node with a
// stub agent: no browser and no Playwright MCP. Gate 13 of scripts/gate.sh runs it, and any
// check that does not hold makes it exit 1. Its first draft was T8's gitignored stub
// (magcoupling-rs/target/gui-smoke-stub.cjs), which only printed what it saw. It checks:
//   - the drag's frame timer, FRAME_TIMER, keeps only the refreshes that draw (plan review 4.1);
//   - the drag's verdict (slice-3 spec 7.2 steps 3 and 4) at its bounds, recorded on every run
//     and failing the run only with frame_check: true;
//   - frame_check is true, false or absent, and the calculator's result records it (T8 M-3);
//   - the script checks the calculator's passed again from its agent's fields (T8 M-4);
//   - an agent that returns nothing leaves its step every field its schema requires (T8 M-4);
//   - the run's passed is both steps'.
//
// Usage: node linkage-sim-rs/scripts/gui_smoke_verdict_test.cjs [script]
// With no argument it checks the repository's script, found from this file's place; with one,
// that copy of it instead (say, `git show <commit>:.claude/workflows/gui-smoke.js` in a file).
'use strict'

const fs = require('fs')
const path = require('path')

const SCRIPT = process.argv[2]
  || path.join(__dirname, '..', '..', '.claude', 'workflows', 'gui-smoke.js')
const BODY = fs.readFileSync(SCRIPT, 'utf8')
// The Workflow tool runs a script's body as an async function of `args` and `agent` (top-level
// await and return); `meta` is the tool's to read, a plain constant here.
const AsyncFunction = (async () => {}).constructor
const workflow = new AsyncFunction('args', 'agent',
  BODY.replace(/^export const meta/m, 'const meta'))

let checked = 0
let failed = 0
const check = (name, holds, got) => {
  checked += 1
  if (holds) {
    console.log(`ok - ${name}`)
  } else {
    failed += 1
    console.log(`not ok - ${name}: ${got}`)
  }
}

// ---- The drag's frame timer, FRAME_TIMER (plan review 4.1): eframe asks for an animation frame
// at every display refresh and draws only when something changed, so the timer keeps a refresh
// only if it made a WebGL draw call.

// Display refreshes in a stand-in window, one per entry of `draws` (true: the refresh makes the
// draw call `call` on a `context`, a WebGL2RenderingContext or a WebGLRenderingContext). Returns
// how many refreshes the timer kept.
const keptOf = (timer, draws, how = {}) => {
  const { context = 'WebGL2RenderingContext', call = 'drawElements' } = how
  const queue = []
  const Context = function () {}
  Context.prototype.drawElements = function () {}
  Context.prototype.drawArrays = function () {}
  const window = { requestAnimationFrame: callback => queue.push(callback), [context]: Context }
  new Function('window', 'performance', `(${timer})()`)(window, { now: () => 0 })
  const gl = new Context()
  for (const draw of draws) {
    window.requestAnimationFrame(() => {
      if (draw) gl[call]()
    })
  }
  while (queue.length) queue.shift()(0)
  return window.__frameMs.length
}
// Spec 7.2's first timer, which keeps every refresh: the control, showing the stand-in can tell.
const FIRST_TIMER = '() => { window.__frameMs = []; const raf = window.requestAnimationFrame.bind(window); window.requestAnimationFrame = cb => raf(t => { const s = performance.now(); cb(t); window.__frameMs.push(performance.now() - s); }); }'
{
  const kept = keptOf(FIRST_TIMER, [false, true])
  check('the first timer, the control, keeps an idle refresh and a drawn one', kept === 2,
    `kept ${kept} of 2`)
}
const timerLine = /^const FRAME_TIMER = `([^`]*)`$/m.exec(BODY)
if (timerLine === null) {
  check('the script has FRAME_TIMER on one line', false, 'no FRAME_TIMER line')
} else {
  for (const [name, draws, how] of [
    ['keeps only the refresh that draws', [false, true], {}],
    ['forgets a draw at the next refresh', [true, false], {}],
    ['keeps a refresh that draws with drawArrays', [false, true], { call: 'drawArrays' }],
    ['keeps a refresh that draws on a WebGL 1 context', [false, true],
      { context: 'WebGLRenderingContext' }],
  ]) {
    const kept = keptOf(timerLine[1], draws, how)
    check(`FRAME_TIMER ${name}`, kept === 1, `kept ${kept} of ${draws.length}`)
  }
}

// ---- The workflow's steps, run with a stub agent.

const LINKAGE = 'gui-smoke' // the agent labels: the hub page and the linkage app
const CALCULATOR = 'gui-smoke-magcoupling' // the calculator

// Runs the workflow as the Workflow tool does (its args a JSON string) with a stub agent that
// answers each step, by its label, with a copy of `readings[label]` (none: the agent returned
// nothing). Gives the workflow's result, or the error it threw, and the options each step handed
// its agent.
const run = async (args, readings) => {
  const asked = {}
  const agent = async (prompt, options) => {
    asked[options.label] = options
    const reading = readings[options.label]
    return reading ? JSON.parse(JSON.stringify(reading)) : null
  }
  try {
    return { result: await workflow(JSON.stringify(args), agent), asked }
  } catch (error) {
    return { error, asked }
  }
}
const calculator = (args, reading) => run({ ...args, linkage: false }, { [CALCULATOR]: reading })

// The checks the calculator's prompt names for its passed, typed here as this test's own
// statement of them: a check the script stops judging fails a case below, and one the script's
// schema gains fails every case that should pass until GOOD has it.
const CHECKS = ['canvas_present', 'first_view', 'details_toggle', 'share_link_loaded',
  'sizing_solved', 'design_file_loaded', 'explorer_ready', 'why_link_opened', 'tooltip_plain',
  'dashboard_scrolls', 'torque_card']
// A calculator agent's reading of a good run; each case changes one thing.
const GOOD = {
  passed: true,
  console_errors: [],
  ...Object.fromEntries(CHECKS.map(field => [field, true])),
  drag_moved: true,
  drag_frames: 60,
  drag_frame_ms_median: 4.1,
  drag_frame_ms_p95: 9.8,
  drag_frame_ms_max: 12.0,
  drag_frames_within_60fps: true,
}
const SLOW = { ...GOOD, drag_frame_ms_p95: 21.3 }
const NO_DRAG = { // the handle not found: drag_moved false and the four numbers left out
  ...GOOD,
  drag_moved: false,
  drag_frames: undefined,
  drag_frame_ms_median: undefined,
  drag_frame_ms_p95: undefined,
  drag_frame_ms_max: undefined,
}
// A linkage agent's reading of a good run.
const LINKAGE_GOOD = {
  passed: true, canvas_present: true, console_errors: [], hub_page: true, title_warning: false,
}
const outcome = ({ result, error }) => {
  if (error) return `threw ${error.message}`
  const m = result.magcoupling || {}
  return `passed=${result.passed} within=${m.drag_frames_within_60fps} frame_check=${m.frame_check}`
}

const main = async () => {
  // The drag's verdict (slice-3 spec 7.2 steps 3 and 4): a drag that moved the gap, drew at least
  // 30 frames and had a p95 within one 60 fps frame (16.7 ms), bounds included. The script works
  // it out from the agent's numbers on every run; it fails the run only with frame_check: true.
  // [case, args, the agent's reading, the run's passed, drag_frames_within_60fps]
  const FRAME_CHECK = { frame_check: true }
  for (const [name, args, reading, passed, within] of [
    ['a fast drag, frame_check', FRAME_CHECK, GOOD, true, true],
    ['a slow drag, frame_check', FRAME_CHECK, SLOW, false, false],
    ['a slow drag, an ordinary run', {}, SLOW, true, false],
    ['a slow drag, frame_check false', { frame_check: false }, SLOW, true, false],
    ['12 frames, frame_check', FRAME_CHECK, { ...GOOD, drag_frames: 12 }, false, false],
    ['no p95, frame_check', FRAME_CHECK, { ...GOOD, drag_frame_ms_p95: null }, false, false],
    ['the handle not held, frame_check', FRAME_CHECK, { ...GOOD, drag_moved: false }, false, false],
    ['the handle not found, frame_check', FRAME_CHECK, NO_DRAG, false, false],
    ['the handle not found, an ordinary run', {}, NO_DRAG, true, false],
    ['the agent returned nothing, frame_check', FRAME_CHECK, null, false, false],
    ['a p95 of exactly 16.7 ms, frame_check', FRAME_CHECK,
      { ...GOOD, drag_frame_ms_p95: 16.7 }, true, true],
    ['exactly 30 frames, frame_check', FRAME_CHECK, { ...GOOD, drag_frames: 30 }, true, true],
    ['a p95 of 16.71 ms, frame_check', FRAME_CHECK,
      { ...GOOD, drag_frame_ms_p95: 16.71 }, false, false],
    ['29 frames, frame_check', FRAME_CHECK, { ...GOOD, drag_frames: 29 }, false, false],
    ['30.5 frames, not a count, frame_check', FRAME_CHECK,
      { ...GOOD, drag_frames: 30.5 }, false, false],
    ['a fast drag with torque_card false, frame_check', FRAME_CHECK,
      { ...GOOD, torque_card: false }, false, true],
  ]) {
    const ran = await calculator(args, reading)
    check(`the drag's verdict: ${name}`, !ran.error && ran.result.passed === passed
      && ran.result.magcoupling.passed === passed
      && ran.result.magcoupling.drag_frames_within_60fps === within, outcome(ran))
  }

  // frame_check is true, false or absent (T8 M-3). Any other value stops the run before its first
  // step, rather than read as false and let a slow drag pass.
  for (const [what, value] of [
    ['the string "true"', 'true'],
    ['the number 1', 1],
    ['null', null],
    ['the string "false"', 'false'],
  ]) {
    const ran = await calculator({ frame_check: value }, SLOW)
    check(`frame_check ${what} stops the run before its first step`, ran.error !== undefined
      && ran.error.message.includes('frame_check') && Object.keys(ran.asked).length === 0,
    outcome(ran))
  }
  // The calculator's result records whether the check applied, so a run whose check did not apply
  // (no frame_check, or a key that is not it) says so.
  for (const [what, args, applied, passed] of [
    ['frame_check true', FRAME_CHECK, true, false],
    ['frame_check false', { frame_check: false }, false, true],
    ['no frame_check', {}, false, true],
    ['frameCheck true, which is not the argument', { frameCheck: true }, false, true],
  ]) {
    const ran = await calculator(args, SLOW)
    check(`${what}: a slow drag's result records frame_check ${applied}`, !ran.error
      && ran.result.magcoupling.frame_check === applied && ran.result.passed === passed,
    outcome(ran))
  }

  // The script checks the calculator's passed again from its agent's reading (T8 M-4), rather
  // than take the agent's word: every check true and no console error, beside the agent's own
  // passed (it alone sees that the page loaded). The drag's fields are the frame rule's (above).
  for (const [name, reading] of [
    ...CHECKS.map(field => [`${field} false with the agent's passed true`,
      { ...GOOD, [field]: false }]),
    ['torque_card left out of the reading', { ...GOOD, torque_card: undefined }],
    ["a console error with the agent's passed true",
      { ...GOOD, console_errors: ['TypeError: x is undefined'] }],
    ['console_errors left out of the reading', { ...GOOD, console_errors: undefined }],
    ["the agent's passed false with every check true", { ...GOOD, passed: false }],
  ]) {
    const ran = await calculator({}, reading)
    check(`the calculator's checks: ${name} fails the run`, !ran.error
      && ran.result.passed === false && ran.result.magcoupling.passed === false, outcome(ran))
  }

  // An agent that returns nothing (an agent error): its step reads every field the step's schema
  // requires, each check false and each list empty, and the drag's numbers stay absent, since the
  // drag did not run.
  const filled = (step, reading, schema) => {
    for (const field of schema.required) {
      const { type } = schema.properties[field]
      const value = reading[field]
      const empty = type === 'boolean'
        ? value === false
        : type === 'array' && Array.isArray(value) && value.length === 0
      check(`${step} with no reading: ${field} is ${type === 'array' ? 'empty' : 'false'}`, empty,
        JSON.stringify(value))
    }
    check(`${step} with no reading: its notes say the agent returned nothing`,
      typeof reading.notes === 'string' && reading.notes.includes('no result'),
      JSON.stringify(reading.notes))
  }
  {
    const ran = await calculator({}, null)
    filled('the calculator', ran.result.magcoupling, ran.asked[CALCULATOR].schema)
    const numbers = ['drag_frames', 'drag_frame_ms_median', 'drag_frame_ms_p95',
      'drag_frame_ms_max'].filter(field => field in ran.result.magcoupling)
    check("the calculator with no reading: the drag's numbers are absent", numbers.length === 0,
      numbers.join(', '))
  }
  {
    const ran = await run({ magcoupling: false }, {})
    filled('the linkage step', ran.result.linkage, ran.asked[LINKAGE].schema)
  }

  // The run's passed is both steps'; the linkage step's fields also sit at the top level.
  for (const [name, readings, passed] of [
    ['both steps pass', { [LINKAGE]: LINKAGE_GOOD, [CALCULATOR]: GOOD }, true],
    ["the linkage step's agent says it failed",
      { [LINKAGE]: { ...LINKAGE_GOOD, passed: false }, [CALCULATOR]: GOOD }, false],
    ['the calculator fails a check',
      { [LINKAGE]: LINKAGE_GOOD, [CALCULATOR]: { ...GOOD, torque_card: false } }, false],
    ["the linkage step's agent returned nothing", { [CALCULATOR]: GOOD }, false],
  ]) {
    const ran = await run({}, readings)
    check(`the run: ${name}`, !ran.error && ran.result.passed === passed
      && ran.result.linkage !== null && ran.result.hub_page === ran.result.linkage.hub_page,
    outcome(ran))
  }

  console.log(failed === 0
    ? `all ${checked} checks hold`
    : `${failed} of ${checked} checks fail`)
  process.exitCode = failed === 0 ? 0 : 1
}

main().catch(error => {
  console.log(`not ok - the test itself failed: ${error.stack}`)
  process.exitCode = 1
})
