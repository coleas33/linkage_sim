//! Tracing between the inputs and the results (decision O-8): a click on an input's label
//! traces the results its value flows into (the registry's `downstream`), a click on a result
//! (any readout: a dashboard row, a table row, a callout) traces the inputs it reads (its
//! `upstream_inputs`). The panel frames every traced path where it is drawn, as the Equation
//! panel frames an equation's terms, in the selection colour (the colour of the frame a leaf
//! term draws around its input row), under the equation's own term colours. The registry knows
//! the dependencies of the results it explains (the A-3 scope: 392 of the 1086 results), so a
//! result without an equation record traces nothing, and an input whose results have none (the
//! drive torque, the adhesive's inputs) reaches no result: its banner says so.

use std::collections::BTreeSet;

use crate::gui::explorer::term_label;
use crate::gui::readouts::registry;

/// The start of the trace's banner over the inputs side.
pub const TRACING: &str = "Tracing";

/// The banner's button that ends the trace.
pub const CLEAR_TRACE: &str = "Clear trace";

/// What is traced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TraceKind {
    /// An input: the trace marks the explained results downstream of it.
    Input,
    /// A result with an equation record: the trace marks the inputs upstream of it.
    Result,
}

/// A trace: its source and the paths it reaches.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Trace {
    pub source: String,
    pub kind: TraceKind,
    /// An input's explained results downstream, or a result's inputs upstream (transitively).
    pub paths: BTreeSet<String>,
}

impl Trace {
    /// The trace of `path`: an input's downstream results, an explained result's upstream
    /// inputs; `None` for a result without an equation record or a path that is neither.
    pub fn of(path: &str) -> Option<Trace> {
        let registry = registry();
        if registry.is_leaf_input(path) {
            return Some(Trace {
                source: path.to_owned(),
                kind: TraceKind::Input,
                paths: registry.downstream(path),
            });
        }
        registry.upstream_inputs(path).map(|inputs| Trace {
            source: path.to_owned(),
            kind: TraceKind::Result,
            paths: inputs.clone(),
        })
    }

    /// Whether the trace marks `path`: its source, or a path it reaches.
    pub fn marks(&self, path: &str) -> bool {
        self.source == path || self.paths.contains(path)
    }

    /// How many of `paths` the trace marks: what a group heading counts.
    pub fn count_in<'a>(&self, paths: impl IntoIterator<Item = &'a str>) -> usize {
        paths.into_iter().filter(|path| self.marks(path)).count()
    }

    /// The banner over the inputs side: what is traced and how far it reaches.
    pub fn banner(&self) -> String {
        let label = term_label(&self.source);
        let count = self.paths.len();
        match self.kind {
            TraceKind::Input if count == 0 => format!(
                "{TRACING} {label}: no explained result reads it (its results have no equation record)"
            ),
            TraceKind::Input => format!("{TRACING} {label}: drives {count} explained results"),
            TraceKind::Result => format!("{TRACING} {label}: reads {count} inputs"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gui::results_table::table_entries;

    #[test]
    fn an_input_traces_the_results_its_value_flows_into() {
        let trace = Trace::of("metal.face_gap_mm").expect("an input");
        assert_eq!(trace.kind, TraceKind::Input);
        assert_eq!(trace.source, "metal.face_gap_mm");
        assert_eq!(trace.paths, registry().downstream("metal.face_gap_mm"));
        // The pull-out reads the gap; the clamp's preload does not.
        assert!(trace.marks("model.pullout_Nm"));
        assert!(!trace.marks("clamps.joint_preload_N"));
        // The source is marked too: its own row is framed.
        assert!(trace.marks("metal.face_gap_mm"));
        assert_eq!(
            trace.banner(),
            format!(
                "{TRACING} Candidate flat-face magnetic gap: drives {} explained results",
                trace.paths.len()
            )
        );
    }

    #[test]
    fn a_result_traces_the_inputs_it_reads() {
        let trace = Trace::of("model.pullout_Nm").expect("an explained result");
        assert_eq!(trace.kind, TraceKind::Result);
        let inputs = registry().upstream_inputs("model.pullout_Nm").unwrap();
        assert_eq!(&trace.paths, inputs);
        assert!(trace.marks("metal.face_gap_mm"));
        assert!(trace.marks("model.pullout_Nm"));
        assert!(!trace.marks("clamps.friction"));
        // Every traced path is an input.
        assert!(trace.paths.iter().all(|p| registry().is_leaf_input(p)));
        assert_eq!(
            trace.banner(),
            format!(
                "{TRACING} Pull-out torque at operating temperature: reads {} inputs",
                inputs.len()
            )
        );
        // Both directions agree: the gap drives the pull-out, which reads the gap.
        assert!(
            Trace::of("metal.face_gap_mm")
                .unwrap()
                .marks("model.pullout_Nm")
        );
    }

    #[test]
    fn an_input_no_explained_result_reads_traces_none_and_says_so() {
        // The drive torque sets the required floor and the verdict, which have no equation
        // record: its trace reaches no result, and its banner says why.
        let trace = Trace::of("coupling.drive_torque_Nm").expect("an input");
        assert_eq!(trace.kind, TraceKind::Input);
        assert!(trace.paths.is_empty());
        assert_eq!(
            trace.banner(),
            format!(
                "{TRACING} Torque the coupling must carry for driving (at the wheel): no \
                 explained result reads it (its results have no equation record)"
            )
        );
        // Only the explained results are traced: those with an equation record.
        let explained = table_entries()
            .iter()
            .filter(|e| registry().equation_for(&e.path).is_some())
            .count();
        assert_eq!((explained, table_entries().len()), (392, 1086));
        // Every input is a leaf of the registry: a click on any input's label traces it.
        for entry in crate::gui::inputs::InputCatalogue::get().all() {
            let trace = Trace::of(&entry.path).unwrap_or_else(|| panic!("{}", entry.path));
            assert_eq!(trace.kind, TraceKind::Input, "{}", entry.path);
        }
    }

    #[test]
    fn a_trace_counts_the_paths_it_marks() {
        let trace = Trace::of("metal.face_gap_mm").unwrap();
        // Its source, a result it drives; not a result it does not, nor an unknown path.
        assert_eq!(
            trace.count_in([
                "metal.face_gap_mm",
                "model.pullout_Nm",
                "clamps.joint_preload_N",
                "no.such.path"
            ]),
            2
        );
        assert_eq!(trace.count_in(std::iter::empty()), 0);
    }

    #[test]
    fn a_result_without_an_equation_record_and_an_unknown_path_trace_nothing() {
        let cell_only = table_entries()
            .iter()
            .find(|e| registry().equation_for(&e.path).is_none())
            .expect("a result without a record");
        assert_eq!(Trace::of(&cell_only.path), None, "{}", cell_only.path);
        assert_eq!(Trace::of("no.such.path"), None);
    }
}
