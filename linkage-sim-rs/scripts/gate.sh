#!/usr/bin/env bash
# Verification gate for the agentic fix loop and the magcoupling port.
# Usage: scripts/gate.sh [--full]   (--full also runs the release WASM builds of both bundles)
# Exit 0 + "GATE PASS" = all gates green. Any failure exits non-zero.
#
# Gates 1-3: linkage-sim-rs (which depends on magcoupling-rs with its feature
# gui: the calculator window, Tools -> Magnetic coupling). Gates 4-6:
# magcoupling-rs, a separate crate (not a workspace member), the engine alone
# (no features); its clippy runs with warnings as errors, as in every
# magcoupling-rs gate. Gates 7-11: magcoupling-rs with its gui and app
# features (the panel and the standalone app): their tests, clippy on native
# and wasm32, the guard that no shipped build (the calculator's own and the
# linkage app's, native and wasm32) has the test-only workbook-parity feature
# (with negative controls that prove the guard trips), and the check that both
# crates lock the same egui, egui_plot, eframe and wasm-bindgen (the CLI
# version deploy-web.yml installs).
# Gate 12: the vendored Python oracle, reference/magcoupling-py: its parity
# suite, and a check that the committed differential test data is current.
# Gate 12 needs a Python with the oracle's dependencies; see oracle_python
# below. Without one it prints a SKIP line and does not fail.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$(cd .. && pwd)"
MAGCOUPLING="$REPO_ROOT/magcoupling-rs/Cargo.toml"
ORACLE="$REPO_ROOT/reference/magcoupling-py"
# MAGCOUPLING_WEB_ARGS, MAGCOUPLING_NATIVE_ARGS, magcoupling_assert_shipped.
# shellcheck source=magcoupling_shipped.sh
source scripts/magcoupling_shipped.sh

# Fails the gate unless the workbook-parity guard trips on the shipped build of
# binary $1 with feature $2 forced on; the remaining arguments are the build's
# cargo check arguments. Cargo's output is captured first, so a build that
# fails to compile fails the gate here instead of passing as a trip.
assert_guard_trips() {
  local bin="$1" feature="$2" forced
  shift 2
  forced="$(cargo check "$@" --features "$feature" --message-format=json-render-diagnostics)"
  if magcoupling_assert_shipped "$bin" <<<"$forced" 2>/dev/null; then
    echo "FAIL gate 10/12: the guard did not trip on $bin with workbook-parity forced on"
    exit 1
  fi
  echo "negative control: the guard trips on $bin when workbook-parity is forced on"
}

# The versions of package $2 in the Cargo.lock $1, one per line.
lock_versions() {
  awk -v want="name = \"$2\"" '$0 == want { getline; gsub(/^version = "|"$/, ""); print }' "$1"
}

# Python for the oracle, in order: $MAGCOUPLING_PYTHON (used as given, so a
# wrong value fails the gate), this checkout's reference/magcoupling-py/.venv,
# then that .venv in another worktree of this repository (it is gitignored, so
# worktrees of one clone usually share a single venv). Prints nothing if none.
oracle_python() {
  if [[ -n "${MAGCOUPLING_PYTHON:-}" ]]; then
    echo "$MAGCOUPLING_PYTHON"
    return
  fi
  local roots=("$REPO_ROOT") line root candidate
  while IFS= read -r line; do
    [[ "$line" == worktree\ * ]] && roots+=("${line#worktree }")
  done < <(git -C "$REPO_ROOT" worktree list --porcelain 2>/dev/null || true)
  for root in "${roots[@]}"; do
    for candidate in "$root/reference/magcoupling-py/.venv/Scripts/python.exe" \
                     "$root/reference/magcoupling-py/.venv/bin/python"; do
      if [[ -x "$candidate" ]]; then
        echo "$candidate"
        return
      fi
    done
  done
}

echo "== gate 1/12: cargo test --all (linkage-sim-rs) =="
cargo test --all

echo "== gate 2/12: cargo clippy --all-targets (linkage-sim-rs) =="
cargo clippy --all-targets

echo "== gate 3/12: WASM check (linkage-sim-rs) =="
cargo check "${LINKAGE_WEB_ARGS[@]}"

echo "== gate 4/12: cargo test (magcoupling-rs) =="
cargo test --manifest-path "$MAGCOUPLING"

echo "== gate 5/12: cargo clippy --all-targets, warnings as errors (magcoupling-rs) =="
cargo clippy --manifest-path "$MAGCOUPLING" --all-targets -- -D warnings

echo "== gate 6/12: WASM check (magcoupling-rs) =="
cargo check --manifest-path "$MAGCOUPLING" --target wasm32-unknown-unknown --lib

echo "== gate 7/12: cargo test --features app (magcoupling-rs panel and app, headless egui; bins build with workbook-parity unified) =="
cargo test --manifest-path "$MAGCOUPLING" --features app

echo "== gate 8/12: cargo clippy --all-targets --features app, warnings as errors (magcoupling-rs) =="
cargo clippy --manifest-path "$MAGCOUPLING" --all-targets --features app -- -D warnings

echo "== gate 9/12: WASM clippy, warnings as errors (magcoupling-rs gui panel; app web entry) =="
cargo clippy --manifest-path "$MAGCOUPLING" --target wasm32-unknown-unknown --features gui --lib -- -D warnings
cargo clippy --manifest-path "$MAGCOUPLING" "${MAGCOUPLING_WEB_ARGS[@]}" -- -D warnings

echo "== gate 10/12: workbook-parity guard (shipped native and wasm32 builds of both apps; negative controls) =="
cargo check --manifest-path "$MAGCOUPLING" "${MAGCOUPLING_NATIVE_ARGS[@]}" --message-format=json-render-diagnostics \
  | magcoupling_assert_shipped magcoupling-app
cargo check --manifest-path "$MAGCOUPLING" "${MAGCOUPLING_WEB_ARGS[@]}" --message-format=json-render-diagnostics \
  | magcoupling_assert_shipped magcoupling-web
# The linkage app's builds carry the calculator's panel (magcoupling-rs, feature gui).
cargo check "${LINKAGE_NATIVE_ARGS[@]}" --message-format=json-render-diagnostics \
  | magcoupling_assert_shipped linkage-gui
cargo check "${LINKAGE_WEB_ARGS[@]}" --message-format=json-render-diagnostics \
  | magcoupling_assert_shipped linkage-web
# Negative controls, one per shipped build: the same build with the feature forced
# on must trip the guard, or the guard has stopped seeing what cargo builds.
assert_guard_trips magcoupling-app workbook-parity --manifest-path "$MAGCOUPLING" "${MAGCOUPLING_NATIVE_ARGS[@]}"
assert_guard_trips magcoupling-web workbook-parity --manifest-path "$MAGCOUPLING" "${MAGCOUPLING_WEB_ARGS[@]}"
assert_guard_trips linkage-gui magcoupling-rs/workbook-parity "${LINKAGE_NATIVE_ARGS[@]}"
assert_guard_trips linkage-web magcoupling-rs/workbook-parity "${LINKAGE_WEB_ARGS[@]}"

echo "== gate 11/12: lock parity (egui, egui_plot, eframe, wasm-bindgen; wasm-bindgen-cli pin in deploy-web.yml) =="
LINKAGE_LOCK="Cargo.lock"
MAGCOUPLING_LOCK="$REPO_ROOT/magcoupling-rs/Cargo.lock"
for pkg in egui egui_plot eframe wasm-bindgen; do
  linkage_version="$(lock_versions "$LINKAGE_LOCK" "$pkg")"
  magcoupling_version="$(lock_versions "$MAGCOUPLING_LOCK" "$pkg")"
  if [[ -z "$linkage_version" || "$linkage_version" != "$magcoupling_version" ]]; then
    echo "FAIL gate 11/12: $pkg is locked at '$linkage_version' in linkage-sim-rs and '$magcoupling_version' in magcoupling-rs"
    exit 1
  fi
  echo "$pkg $linkage_version in both lock files"
done
CLI_PIN="$(grep -oE 'wasm-bindgen-cli@[0-9.]+' "$REPO_ROOT/.github/workflows/deploy-web.yml" | head -n 1 | cut -d@ -f2)"
if [[ "$CLI_PIN" != "$(lock_versions "$MAGCOUPLING_LOCK" wasm-bindgen)" ]]; then
  echo "FAIL gate 11/12: deploy-web.yml installs wasm-bindgen-cli '$CLI_PIN', the lock files have wasm-bindgen $(lock_versions "$MAGCOUPLING_LOCK" wasm-bindgen)"
  exit 1
fi
echo "deploy-web.yml installs wasm-bindgen-cli $CLI_PIN"

echo "== gate 12/12: vendored Python oracle (parity suite, differential data freshness) =="
PY="$(oracle_python)"
if [[ -z "$PY" ]]; then
  echo "SKIP gate 12/12: no Python for reference/magcoupling-py (create its .venv as in docs/ai/03-structure.yaml, or set MAGCOUPLING_PYTHON)"
else
  echo "oracle Python: $PY"
  (cd "$ORACLE" && "$PY" -m pytest -q tests -p no:cacheprovider)
  (cd "$ORACLE" && "$PY" tools/gen_differential.py --check)
fi

if [[ "${1:-}" == "--full" ]]; then
  echo "== gate 13 (--full): release WASM builds (linkage and magcoupling bundles) =="
  bash scripts/build_web.sh
fi

echo "GATE PASS"
