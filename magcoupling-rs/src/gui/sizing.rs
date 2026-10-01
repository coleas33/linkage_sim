//! The sizing mode of the Key design group (spec Addendum A1): Magnets → Torque, the forward
//! calculation, or Torque → Magnets, inverse sizing by [`crate::engine::sizing::solve`].
//!
//! The mode, the free variable and the target torque are GUI state (A-2 decision A2-9): the
//! engine takes them as arguments. Design files and share links record them (decision M41-4).

use crate::engine::sizing::FreeVariable;

/// Which way the calculator runs (spec A1 "Mode switch").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SizingMode {
    /// The forward calculation: the inputs give the torque.
    MagnetsToTorque,
    /// Inverse sizing: the target torque gives the free variable.
    TorqueToMagnets,
}

impl SizingMode {
    /// Both modes; the first is the default.
    pub const ALL: [SizingMode; 2] = [SizingMode::MagnetsToTorque, SizingMode::TorqueToMagnets];

    /// The name a design file and a share link record.
    pub const fn key(self) -> &'static str {
        match self {
            SizingMode::MagnetsToTorque => "magnets_to_torque",
            SizingMode::TorqueToMagnets => "torque_to_magnets",
        }
    }

    /// The text of the mode switch: ASCII arrows, since egui's default fonts have no U+2192.
    pub const fn label(self) -> &'static str {
        match self {
            SizingMode::MagnetsToTorque => "Magnets -> Torque",
            SizingMode::TorqueToMagnets => "Torque -> Magnets",
        }
    }

    /// The mode a design file names, if it names one.
    pub fn from_key(key: &str) -> Option<SizingMode> {
        SizingMode::ALL.into_iter().find(|mode| mode.key() == key)
    }
}

/// The name a design file and a share link record for a free variable.
pub const fn variable_key(variable: FreeVariable) -> &'static str {
    match variable {
        FreeVariable::AxialLength => "axial_length",
        FreeVariable::MagnetsPerRing => "magnets_per_ring",
        FreeVariable::RingRadius => "ring_radius",
    }
}

/// The free variable a design file names, if it names one.
pub fn variable_from_key(key: &str) -> Option<FreeVariable> {
    FreeVariable::ALL
        .into_iter()
        .find(|&variable| variable_key(variable) == key)
}

/// The text of the free-variable picker.
pub const fn variable_label(variable: FreeVariable) -> &'static str {
    match variable {
        FreeVariable::AxialLength => "Axial magnet length",
        FreeVariable::MagnetsPerRing => "Magnets per ring",
        FreeVariable::RingRadius => "Ring radius",
    }
}

/// The input whose metadata gives the target torque its unit, range and step: the hot
/// minimum requirement, the torque inverse sizing is usually asked to meet.
pub const TARGET_RANGE_INPUT: &str = "metal.required_min_Nm";

/// The sizing state: the mode, the free variable and the target torque.
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(non_snake_case)] // unit suffix, as the engine's names
pub struct SizingState {
    pub mode: SizingMode,
    pub variable: FreeVariable,
    /// The hot-low torque with production variation that inverse sizing must reach [N·m].
    pub target_Nm: f64,
}

/// The default target torque [N·m]: the workbook's hot minimum (`metal.required_min_Nm`,
/// Metal design C7), decision M41-9.
pub const DEFAULT_TARGET_NM: f64 = 2.5;

impl Default for SizingState {
    fn default() -> Self {
        Self {
            mode: SizingMode::MagnetsToTorque,
            variable: FreeVariable::ALL[0],
            target_Nm: DEFAULT_TARGET_NM,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DesignInputs;

    #[test]
    fn every_mode_and_free_variable_round_trips_through_its_key() {
        for mode in SizingMode::ALL {
            assert_eq!(SizingMode::from_key(mode.key()), Some(mode));
        }
        for variable in FreeVariable::ALL {
            assert_eq!(variable_from_key(variable_key(variable)), Some(variable));
        }
        assert_eq!(SizingMode::from_key("Torque"), None);
        assert_eq!(variable_from_key(""), None);
    }

    #[test]
    fn the_default_state_is_the_forward_calculation_at_the_hot_minimum() {
        let state = SizingState::default();
        assert_eq!(state.mode, SizingMode::MagnetsToTorque);
        assert_eq!(state.variable, FreeVariable::AxialLength);
        assert_eq!(
            state.target_Nm,
            DesignInputs::default().metal.required_min_Nm
        );
    }
}
