#!/usr/bin/env bash
# Verification gate for the agentic fix loop and the magcoupling port.
# Usage: scripts/gate.sh [--full]   (--full also runs the release WASM build)
# Exit 0 + "GATE PASS" = all gates green. Any failure exits non-zero.
#
# Gates 1-3: linkage-sim-rs. Gates 4-6: magcoupling-rs, a separate crate (not
# a workspace member); its clippy runs with warnings as errors. Gate 7: the
# vendored Python oracle, reference/magcoupling-py: its parity suite, and a
# check that the committed differential test data is current. Gate 7 needs a
# Python with the oracle's dependencies; see oracle_python below. Without one
# it prints a SKIP line and does not fail.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$(cd .. && pwd)"
MAGCOUPLING="$REPO_ROOT/magcoupling-rs/Cargo.toml"
ORACLE="$REPO_ROOT/reference/magcoupling-py"

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

echo "== gate 1/7: cargo test --all (linkage-sim-rs) =="
cargo test --all

echo "== gate 2/7: cargo clippy --all-targets (linkage-sim-rs) =="
cargo clippy --all-targets

echo "== gate 3/7: WASM check (linkage-sim-rs) =="
cargo check --target wasm32-unknown-unknown --bin linkage-web --no-default-features --features raster

echo "== gate 4/7: cargo test (magcoupling-rs) =="
cargo test --manifest-path "$MAGCOUPLING"

echo "== gate 5/7: cargo clippy --all-targets, warnings as errors (magcoupling-rs) =="
cargo clippy --manifest-path "$MAGCOUPLING" --all-targets -- -D warnings

echo "== gate 6/7: WASM check (magcoupling-rs) =="
cargo check --manifest-path "$MAGCOUPLING" --target wasm32-unknown-unknown --lib

echo "== gate 7/7: vendored Python oracle (parity suite, differential data freshness) =="
PY="$(oracle_python)"
if [[ -z "$PY" ]]; then
  echo "SKIP gate 7/7: no Python for reference/magcoupling-py (create its .venv as in docs/ai/03-structure.yaml, or set MAGCOUPLING_PYTHON)"
else
  echo "oracle Python: $PY"
  (cd "$ORACLE" && "$PY" -m pytest -q tests -p no:cacheprovider)
  (cd "$ORACLE" && "$PY" tools/gen_differential.py --check)
fi

if [[ "${1:-}" == "--full" ]]; then
  echo "== gate 8 (--full): release WASM build =="
  ./scripts/build_web.sh
fi

echo "GATE PASS"
