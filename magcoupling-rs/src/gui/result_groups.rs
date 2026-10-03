//! The groups of the results table (decision O-6): the headline first (the dashboard's rows,
//! [`DASHBOARD`]), then the physics chains: the A-3 chains of the explorer's scope ([`SCOPE`]:
//! torque, temperature, demagnetization, slip heating, clamps, geometry) and two of results the
//! scope leaves out (adhesive, mass), each chain also taking the results of its nested groups
//! ([`CHAIN_PREFIXES`]: the cold demagnetization limits, the slip temperatures, the adhesive and
//! bond screens); then every other result under its package group, the "Other results". Every
//! result is in exactly one group: a path on the dashboard and in a chain, or in two chains, sits
//! in the first group that lists it.
//!
//! The groups index [`table_entries`], built once; the CSV and JSON exports keep the engine's
//! order.

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::engine::explain::scope::SCOPE;
use crate::gui::dashboard::DASHBOARD;
use crate::gui::results_table::{TableEntry, table_entries};

/// The headline group's heading.
pub const HEADLINE_LABEL: &str = "Headline";

/// The heading over the package groups of the results no chain lists.
pub const OTHER_RESULTS: &str = "Other results";

/// The chain groups, in the order shown, by id, with their headings: the chains of the
/// explorer's scope in its order (its `dashboard` entry is the headline's), with the adhesive
/// after slip heating and the mass last. `adhesive` and `mass` are no chain of the scope: only
/// [`CHAIN_PREFIXES`] fills them.
pub const CHAIN_LABELS: [(&str, &str); 8] = [
    ("torque", "Torque"),
    ("temperature", "Temperature"),
    ("demagnetization", "Demagnetization"),
    ("slip_heating", "Slip heating"),
    ("adhesive", "Adhesive"),
    ("clamps", "Clamps"),
    ("geometry", "Geometry"),
    ("mass", "Mass"),
];

/// The chain each result the scope leaves out goes to, by path prefix, before the other results
/// (decision O-6): a nested group of a package (`temperature.demag.`), the stem of two
/// (`temperature.adhesive` takes `temperature.adhesive.` and `temperature.adhesive_life.`), or
/// one result's whole path (`metal.corner_gap_mm`). A result goes to the chain of the first
/// entry its path starts with, after the chain's own rows, in schema order; a test checks that
/// every entry places a result.
pub const CHAIN_PREFIXES: [(&str, &str); 19] = [
    ("temperature.demag.", "demagnetization"),
    // The summary rows the temperature chain leaves out: the steady temperatures while slipping,
    // the peak with the slip fault, the slip rotations and the average slip heating over life.
    ("temperature.summary.", "slip_heating"),
    ("temperature.magnet_life.", "temperature"),
    ("temperature.thermal.", "slip_heating"),
    ("temperature.slip_life.", "slip_heating"),
    ("temperature.duty.", "slip_heating"),
    ("temperature.adhesive", "adhesive"),
    ("temperature.mismatch.", "adhesive"),
    ("mass.", "mass"),
    // The clearances, gaps, diameters and reserves of the metal design.
    ("metal.running_clearance_mm", "geometry"),
    ("metal.corner_gap_mm", "geometry"),
    ("metal.assembled_face_gap_mm", "geometry"),
    ("metal.corner_clearance_mm", "geometry"),
    ("metal.nominal_sleeve_liner_mm", "geometry"),
    ("metal.allowed_radial_disp_mm", "geometry"),
    ("metal.cup_body_od_mm", "geometry"),
    ("metal.diameter_reserve_mm", "geometry"),
    ("metal.large_dia_reserve_mm", "geometry"),
    ("metal.axial_reserve_mm", "geometry"),
];

/// The heading of each package group of the other results, by the first segment of the path
/// (a table's index dropped: `gap_sweep[3].f_end` is in `gap_sweep`), in schema order.
pub const PACKAGE_LABELS: [(&str, &str); 12] = [
    ("calibration", "Calibration"),
    ("model", "Coupling model"),
    ("mass", "Mass"),
    ("retainers", "Retainers"),
    ("metal", "Metal design"),
    ("materials", "Materials"),
    ("temperature", "Temperature design"),
    ("clamps", "Shaft clamps"),
    ("warnings", "Material warnings"),
    ("housing", "Housing"),
    ("gap_sweep", "Gap sweep"),
    ("pole_sweep", "Pole sweep"),
];

/// One group of the results table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResultGroup {
    /// `headline`, a chain's id, or a package's (`calibration`, `gap_sweep`). A chain and a
    /// package may share an id (`temperature`, `clamps`): `other` tells them apart.
    pub id: &'static str,
    pub label: &'static str,
    /// Under the "Other results" heading: a package group.
    pub other: bool,
    /// Its rows, as indices into [`table_entries`], in the order shown.
    pub rows: Vec<usize>,
}

/// A path with each table index emptied, as the explorer's scope writes a table's every row:
/// `clamps.table[3].preload_N` gives `clamps.table[].preload_N`; a path without an index is
/// unchanged.
pub fn row_template(path: &str) -> String {
    let mut template = String::with_capacity(path.len());
    let mut in_index = false;
    for c in path.chars() {
        match c {
            '[' => {
                in_index = true;
                template.push(c);
            }
            ']' => {
                in_index = false;
                template.push(c);
            }
            _ if in_index => {}
            _ => template.push(c),
        }
    }
    template
}

/// The package of a path: its first segment, a table's index dropped.
pub fn package_of(path: &str) -> &str {
    let first = path.split('.').next().unwrap_or(path);
    first.split('[').next().unwrap_or(first)
}

/// The heading of a package; `None` if [`PACKAGE_LABELS`] lacks it (a test checks none does).
pub fn package_label(package: &str) -> Option<&'static str> {
    PACKAGE_LABELS
        .iter()
        .find(|(id, _)| *id == package)
        .map(|(_, label)| *label)
}

/// The groups of `entries`: the headline, the chains (each its scope rows, then the rows its
/// [`CHAIN_PREFIXES`] place), then the other results by package; a group left empty (every row
/// in an earlier group) is dropped.
pub fn groups_of(entries: &[TableEntry]) -> Vec<ResultGroup> {
    let mut by_template: HashMap<String, Vec<usize>> = HashMap::new();
    for (index, entry) in entries.iter().enumerate() {
        by_template
            .entry(row_template(&entry.path))
            .or_default()
            .push(index);
    }
    let mut placed = vec![false; entries.len()];
    // The rows of `paths` (a table's template gives each of its rows) not placed yet, in order.
    let mut take = |paths: &mut dyn Iterator<Item = &str>| -> Vec<usize> {
        let mut rows = Vec::new();
        for path in paths {
            for &index in by_template.get(path).map_or(&[][..], Vec::as_slice) {
                if !placed[index] {
                    placed[index] = true;
                    rows.push(index);
                }
            }
        }
        rows
    };
    let mut groups = vec![ResultGroup {
        id: "headline",
        label: HEADLINE_LABEL,
        other: false,
        rows: take(&mut DASHBOARD.iter().map(|(path, _)| *path)),
    }];
    for (id, label) in CHAIN_LABELS {
        // A chain of the scope starts with its rows; the adhesive and the mass start empty.
        let rows = match SCOPE.iter().find(|chain| chain.id == id) {
            Some(chain) => take(&mut chain.paths.iter().copied()),
            None => Vec::new(),
        };
        groups.push(ResultGroup {
            id,
            label,
            other: false,
            rows,
        });
    }
    for (index, entry) in entries.iter().enumerate() {
        if placed[index] {
            continue;
        }
        let Some(&(_, chain)) = CHAIN_PREFIXES
            .iter()
            .find(|(prefix, _)| entry.path.starts_with(prefix))
        else {
            continue;
        };
        let group = groups
            .iter_mut()
            .find(|group| !group.other && group.id == chain)
            .unwrap_or_else(|| panic!("CHAIN_PREFIXES: no chain {chain} in CHAIN_LABELS"));
        group.rows.push(index);
        placed[index] = true;
    }
    for (id, label) in PACKAGE_LABELS {
        let rows: Vec<usize> = (0..entries.len())
            .filter(|&index| !placed[index] && package_of(&entries[index].path) == id)
            .collect();
        groups.push(ResultGroup {
            id,
            label,
            other: true,
            rows,
        });
    }
    groups.retain(|group| !group.rows.is_empty());
    groups
}

/// The groups of the results table, built once (the layout of the results does not depend on
/// the inputs).
pub fn result_groups() -> &'static [ResultGroup] {
    static GROUPS: OnceLock<Vec<ResultGroup>> = OnceLock::new();
    GROUPS.get_or_init(|| groups_of(table_entries()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The group holding the row of `path`.
    fn group_of(path: &str) -> &'static str {
        let entries = table_entries();
        result_groups()
            .iter()
            .find(|g| g.rows.iter().any(|&i| entries[i].path == path))
            .unwrap_or_else(|| panic!("{path} in no group"))
            .id
    }

    #[test]
    fn every_result_is_in_exactly_one_group() {
        let entries = table_entries();
        let mut rows: Vec<usize> = result_groups()
            .iter()
            .flat_map(|g| g.rows.iter().copied())
            .collect();
        assert_eq!(rows.len(), entries.len(), "no row twice, none left out");
        rows.sort_unstable();
        assert_eq!(rows, (0..entries.len()).collect::<Vec<_>>());
        assert!(result_groups().iter().all(|g| !g.rows.is_empty()));
    }

    #[test]
    fn the_headline_comes_first_then_the_chains_then_the_other_results() {
        let groups = result_groups();
        let ids: Vec<&str> = groups.iter().map(|g| g.id).collect();
        assert_eq!(
            ids[..9],
            [
                "headline",
                "torque",
                "temperature",
                "demagnetization",
                "slip_heating",
                "adhesive",
                "clamps",
                "geometry",
                "mass"
            ]
        );
        assert!(groups[..9].iter().all(|g| !g.other));
        assert!(groups[9..].iter().all(|g| g.other));
        // A group is known by its id and its section.
        let mut keys: Vec<(bool, &str)> = groups.iter().map(|g| (g.other, g.id)).collect();
        keys.sort_unstable();
        keys.dedup();
        assert_eq!(keys.len(), groups.len());
        // The headline is the dashboard's rows, in its order.
        let entries = table_entries();
        let headline: Vec<&str> = groups[0]
            .rows
            .iter()
            .map(|&i| entries[i].path.as_str())
            .collect();
        let dashboard: Vec<&str> = DASHBOARD.iter().map(|(path, _)| *path).collect();
        assert_eq!(headline, dashboard);
        // A path in two places sits in the first: the pull-out on the dashboard and in two
        // chains, an onset in the temperature and the demagnetization chains.
        assert_eq!(group_of("model.pullout_Nm"), "headline");
        assert_eq!(group_of("model.f_end"), "torque");
        assert_eq!(
            group_of("temperature.summary.onset_aligned_C"),
            "temperature"
        );
        assert_eq!(
            group_of("temperature.demag.onset_pullout_C"),
            "demagnetization"
        );
        // A table template places every row of the table (the clamp table has five).
        for row in 0..5 {
            assert_eq!(
                group_of(&format!("clamps.table[{row}].preload_N")),
                "clamps"
            );
        }
        assert_eq!(group_of("clamps.table[0].size"), "clamps");
        // The results of a chain's nested groups that the scope leaves out join the chain.
        assert_eq!(group_of("temperature.demag.cold_check"), "demagnetization");
        assert_eq!(
            group_of("temperature.demag.inner_magnet_limit_C"),
            "demagnetization"
        );
        assert_eq!(
            group_of("temperature.summary.peak_with_fault_C"),
            "slip_heating"
        );
        assert_eq!(
            group_of("temperature.thermal.temp_at_fault_C"),
            "slip_heating"
        );
        assert_eq!(group_of("temperature.magnet_life.peak_C"), "temperature");
        assert_eq!(group_of("temperature.mismatch.reading"), "adhesive");
        assert_eq!(
            group_of("temperature.adhesive_life.daily_screen"),
            "adhesive"
        );
        assert_eq!(group_of("mass.magnets_g"), "mass");
        assert_eq!(group_of("metal.corner_gap_mm"), "geometry");
        // So no temperature design result is left among the other results.
        assert!(!groups.iter().any(|g| g.other && g.id == "temperature"));
        // The rest by package, in schema order.
        assert_eq!(group_of("model.verdict"), "model");
        assert_eq!(group_of("gap_sweep[3].f_end"), "gap_sweep");
        assert_eq!(group_of("retainers.retainers_g"), "retainers");
    }

    #[test]
    fn every_chain_path_names_results_and_every_package_has_a_heading() {
        let entries = table_entries();
        for chain in SCOPE {
            for path in chain.paths {
                assert!(
                    entries.iter().any(|e| row_template(&e.path) == *path),
                    "{}: {path} names no result",
                    chain.id
                );
            }
        }
        // Every scope entry but the dashboard's is a chain group, in the scope's order.
        let scope: Vec<&str> = SCOPE
            .iter()
            .map(|c| c.id)
            .filter(|id| *id != "dashboard")
            .collect();
        let chains: Vec<&str> = CHAIN_LABELS
            .iter()
            .map(|(id, _)| *id)
            .filter(|id| SCOPE.iter().any(|c| c.id == *id))
            .collect();
        assert_eq!(scope, chains);
        for entry in entries {
            let package = package_of(&entry.path);
            assert!(package_label(package).is_some(), "no heading for {package}");
        }
        assert_eq!(package_of("gap_sweep[12].tau_Pa"), "gap_sweep");
        assert_eq!(package_of("model.f_end"), "model");
        assert_eq!(
            row_template("clamps.table[3].preload_N"),
            "clamps.table[].preload_N"
        );
        assert_eq!(row_template("model.f_end"), "model.f_end");
    }

    #[test]
    fn every_chain_prefix_places_a_result_the_scope_leaves_out() {
        let entries = table_entries();
        let groups = result_groups();
        // A result the dashboard or a chain of the scope lists is placed by that list.
        let listed = |path: &str| {
            let template = row_template(path);
            DASHBOARD.iter().any(|(p, _)| *p == template)
                || SCOPE.iter().any(|c| c.paths.contains(&template.as_str()))
        };
        for (prefix, chain) in CHAIN_PREFIXES {
            let group = groups
                .iter()
                .find(|g| !g.other && g.id == chain)
                .unwrap_or_else(|| panic!("{prefix}: no chain group {chain}"));
            let placed = group
                .rows
                .iter()
                .map(|&i| entries[i].path.as_str())
                .filter(|path| !listed(path))
                .filter(|path| {
                    CHAIN_PREFIXES
                        .iter()
                        .find(|(p, _)| path.starts_with(p))
                        .is_some_and(|(p, _)| *p == prefix)
                })
                .count();
            assert!(placed > 0, "{prefix} places no result in {chain}");
        }
        // A chain of no scope entry is filled by its prefixes alone, and has rows.
        for (id, _) in CHAIN_LABELS {
            if SCOPE.iter().all(|c| c.id != id) {
                assert!(CHAIN_PREFIXES.iter().any(|(_, c)| *c == id), "{id}");
                assert!(groups.iter().any(|g| !g.other && g.id == id), "{id}");
            }
        }
    }
}
