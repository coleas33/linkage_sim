//! Time modulation types for external loads.

use meval;
use serde::{Deserialize, Serialize};

/// Time modulation for external loads.
///
/// Multiplies the base force/torque by a time-dependent factor.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "modulation_type")]
pub enum TimeModulation {
    /// Constant (default, no modulation). Factor = 1.0 always.
    Constant,
    /// Sinusoidal: factor = sin(omega * t + phase).
    Sinusoidal { omega: f64, phase: f64 },
    /// Step: zero before t_on, full after t_on.
    Step { t_on: f64 },
    /// Ramp: linearly from 0 to 1 over [t_start, t_end].
    Ramp { t_start: f64, t_end: f64 },
    /// User-defined expression of time: e.g., "sin(2*pi*t)" or "1 - exp(-t/0.5)".
    /// Uses the `meval` crate for parsing and evaluation.
    Expression { expr: String },
}

impl Default for TimeModulation {
    fn default() -> Self {
        TimeModulation::Constant
    }
}

impl TimeModulation {
    /// Compute the modulation factor at time `t`.
    ///
    /// Note: for the `Expression` variant this re-parses the expression string
    /// on every call. In hot loops prefer [`compile`] to pre-parse once.
    pub fn factor(&self, t: f64) -> f64 {
        match self {
            TimeModulation::Constant => 1.0,
            TimeModulation::Sinusoidal { omega, phase } => (omega * t + phase).sin(),
            TimeModulation::Step { t_on } => {
                if t >= *t_on {
                    1.0
                } else {
                    0.0
                }
            }
            TimeModulation::Ramp { t_start, t_end } => {
                if t <= *t_start {
                    0.0
                } else if t >= *t_end {
                    1.0
                } else {
                    (t - t_start) / (t_end - t_start)
                }
            }
            TimeModulation::Expression { expr } => {
                // Parse and evaluate (re-parse each call for thread safety,
                // same pattern as ExprEval in the driver module).
                match expr.parse::<meval::Expr>() {
                    Ok(parsed) => match parsed.bind("t") {
                        Ok(f) => {
                            let val = f(t);
                            if val.is_finite() { val } else { 0.0 }
                        }
                        Err(_) => {
                            log::warn!("TimeModulation: failed to bind variable 't' in expression '{expr}' — force disabled");
                            0.0
                        }
                    },
                    Err(_) => {
                        log::warn!("TimeModulation: failed to parse expression '{expr}' — force disabled");
                        0.0
                    }
                }
            }
        }
    }

    /// Pre-compile this modulation into a closure that can be called repeatedly
    /// without re-parsing expression strings.
    ///
    /// For `Constant`, `Sinusoidal`, `Step`, and `Ramp` variants this simply
    /// captures the parameters. For `Expression` the meval string is parsed
    /// once and the bound closure is captured, eliminating the O(n) parse on
    /// every evaluation.
    ///
    /// The returned closure is **not** `Send`/`Sync` (meval closures aren't),
    /// but that is fine for single-threaded simulation loops.
    pub fn compile(&self) -> Box<dyn Fn(f64) -> f64> {
        match self {
            TimeModulation::Constant => Box::new(|_t| 1.0),
            TimeModulation::Sinusoidal { omega, phase } => {
                let omega = *omega;
                let phase = *phase;
                Box::new(move |t| (omega * t + phase).sin())
            }
            TimeModulation::Step { t_on } => {
                let t_on = *t_on;
                Box::new(move |t| if t >= t_on { 1.0 } else { 0.0 })
            }
            TimeModulation::Ramp { t_start, t_end } => {
                let t_start = *t_start;
                let t_end = *t_end;
                Box::new(move |t| {
                    if t <= t_start {
                        0.0
                    } else if t >= t_end {
                        1.0
                    } else {
                        (t - t_start) / (t_end - t_start)
                    }
                })
            }
            TimeModulation::Expression { expr } => {
                match expr.parse::<meval::Expr>() {
                    Ok(parsed) => match parsed.bind("t") {
                        Ok(f) => Box::new(move |t| {
                            let val = f(t);
                            if val.is_finite() { val } else { 0.0 }
                        }),
                        Err(_) => {
                            log::warn!("TimeModulation::compile: failed to bind 't' in expression '{expr}' — force disabled");
                            Box::new(|_t| 0.0)
                        }
                    },
                    Err(_) => {
                        log::warn!("TimeModulation::compile: failed to parse expression '{expr}' — force disabled");
                        Box::new(|_t| 0.0)
                    }
                }
            }
        }
    }
}
