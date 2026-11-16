//! The one escaper per output context for names that come from a mechanism
//! file (body, joint, attachment-point and trace names). A shared file can name
//! a body anything, so every exporter writes names through these.

/// Text or attribute value in HTML or SVG: `& < > " '`.
pub(super) fn html_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(c),
        }
    }
    out
}

/// A JS string literal, quotes included, for a name inside a `<script>`.
/// serde_json does the quoting; every `<` becomes `\u003c` (the same string to
/// JS) so no name can write `</script>` or `<!--`.
pub(super) fn js_str(s: &str) -> String {
    serde_json::to_string(s).unwrap_or_else(|_| "\"\"".to_string()).replace('<', "\\u003c")
}

/// A name as a DXF text value (a layer name). DXF is line-oriented: a code
/// line, then a value line, so a line break in a name would shift every group
/// after it. Control characters (newline, CR, tab, ...) become `_`.
pub(super) fn dxf_text(s: &str) -> String {
    s.chars().map(|c| if c.is_control() { '_' } else { c }).collect()
}

#[cfg(test)]
pub(super) mod test_support {
    use crate::core::body::{make_bar, make_ground};
    use crate::core::mechanism::Mechanism;

    pub const SCRIPT_NAME: &str = "<script>alert(1)</script>";
    pub const QUOTE_NAME: &str = "x\"</script>";

    /// A four-bar whose bodies and joints carry hostile names.
    pub fn hostile_fourbar() -> (Mechanism, nalgebra::DVector<f64>) {
        let ground = make_ground(&[("O2", 0.0, 0.0), ("O4", 0.038, 0.0)]);
        let crank = make_bar(SCRIPT_NAME, "A", "B", 0.01, 0.0, 0.0);
        let coupler = make_bar(QUOTE_NAME, "B", "C", 0.04, 0.0, 0.0);
        let rocker = make_bar("rocker", "C", "D", 0.03, 0.0, 0.0);
        let mut mech = Mechanism::new();
        mech.add_body(ground).unwrap();
        mech.add_body(crank).unwrap();
        mech.add_body(coupler).unwrap();
        mech.add_body(rocker).unwrap();
        mech.add_revolute_joint("<img src=x onerror=alert(1)>", "ground", "O2", SCRIPT_NAME, "A").unwrap();
        mech.add_revolute_joint(QUOTE_NAME, SCRIPT_NAME, "B", QUOTE_NAME, "B").unwrap();
        mech.add_revolute_joint("J3", QUOTE_NAME, "C", "rocker", "C").unwrap();
        mech.add_revolute_joint("J4", "rocker", "D", "ground", "O4").unwrap();
        mech.add_constant_speed_driver("D1", "ground", SCRIPT_NAME, 2.0 * std::f64::consts::PI, 0.0).unwrap();
        mech.build().unwrap();
        let mut q0 = mech.state().make_q();
        mech.state().set_pose(SCRIPT_NAME, &mut q0, 0.005, 0.0, 0.0);
        mech.state().set_pose(QUOTE_NAME, &mut q0, 0.025, 0.0, 0.0);
        mech.state().set_pose("rocker", &mut q0, 0.04, 0.005, 0.5);
        let q = crate::solver::kinematics::solve_position(&mech, &q0, 0.0, 1e-10, 50).unwrap().q;
        (mech, q)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn html_escape_covers_all_five_characters() {
        assert_eq!(html_escape("<a href=\"x\">&'"), "&lt;a href=&quot;x&quot;&gt;&amp;&#39;");
    }

    #[test]
    fn js_str_quotes_and_cannot_close_the_script() {
        let s = js_str("x\"</script><!--");
        assert_eq!(s, "\"x\\\"\\u003c/script>\\u003c!--\"");
        assert!(!s.contains('<'));
    }

    #[test]
    fn dxf_text_drops_line_breaks() {
        assert_eq!(dxf_text("a\nb\r0\tc"), "a_b_0_c");
    }
}