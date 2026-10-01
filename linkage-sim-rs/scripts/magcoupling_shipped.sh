# Sourced, never run: the cargo arguments of the shipped magcoupling-rs builds,
# defined once, and the guard that no shipped build has the test-only
# workbook-parity feature. Used by build_magcoupling_web.sh (the web bundle
# that deploy-web.yml ships) and gate.sh.
#
# Why a guard on cargo's output and not a compile_error!: workbook-parity
# reaches cargo test and cargo clippy --all-targets through the self
# dev-dependency in magcoupling-rs/Cargo.toml, and cargo unifies
# dev-dependency features across every unit of a test build, the binaries
# included. A compile_error! on app + workbook-parity would therefore break
# cargo test --features app. Instead the guard reads cargo's own record of each
# unit it compiled (--message-format=json, the "features" of every
# "compiler-artifact" line), so it sees exactly the feature set cargo built
# the shipped binary with.

# The web bundle (linkage-sim-rs/web/magcoupling/, served at /magcoupling/).
MAGCOUPLING_WEB_ARGS=(--features app --bin magcoupling-web --target wasm32-unknown-unknown)
# The native desktop app.
MAGCOUPLING_NATIVE_ARGS=(--features app --bin magcoupling-app)

# Reads cargo's --message-format=json (or json-render-diagnostics) output on
# stdin. Succeeds only if the binary $1 was among the compiled units, at least
# one magcoupling-rs unit was, and no magcoupling-rs unit had workbook-parity.
# Reads all of stdin before deciding, so cargo never sees a closed pipe.
magcoupling_assert_shipped() {
  local bin="$1" artifacts units features
  artifacts="$(grep -F '"reason":"compiler-artifact"' || true)"
  if ! grep -F "\"name\":\"$bin\"" <<<"$artifacts" | grep -qF '"kind":["bin"]'; then
    echo "parity guard: cargo compiled no binary $bin; nothing was checked" >&2
    return 1
  fi
  # Match the package by name, wherever its folder lives: path+file://<dir>#magcoupling-rs@<ver>
  # when the folder name differs, path+file://<dir>/magcoupling-rs#<ver> when it matches.
  units="$(grep -E '"package_id":"path\+file://[^"]*(/magcoupling-rs#|#magcoupling-rs@)' <<<"$artifacts" || true)"
  if [[ -z "$units" ]]; then
    echo "parity guard: cargo compiled no magcoupling-rs unit for $bin; nothing was checked" >&2
    return 1
  fi
  features="$(grep -oE '"features":\[[^]]*\]' <<<"$units" | sort -u)"
  if grep -qF '"workbook-parity"' <<<"$features"; then
    echo "parity guard: FAIL: $bin was built with the test-only workbook-parity feature" >&2
    echo "magcoupling-rs features: $features" >&2
    return 1
  fi
  echo "parity guard: $bin builds without workbook-parity; magcoupling-rs features: $features"
}
