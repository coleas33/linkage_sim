# reference/

Upstream code kept verbatim as an **oracle** for tests.

- `magcoupling-py/`: the magnetic coupling calculator (Python port of
  `magnetic_coupling_torque_calculator.xlsx`, version 1.0.0). **Do not edit
  `magcoupling-py/magcoupling/`**: parity and differential tests for the Rust
  port (`magcoupling-rs/`) compare against it unchanged. Independent physics
  checks live in `magcoupling-py/audit/`
  (spec: `docs/superpowers/specs/2026-09-28-magcoupling-calculator-design.md`).
