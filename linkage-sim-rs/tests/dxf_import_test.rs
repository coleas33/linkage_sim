/// Integration test: write a minimal DXF file, parse it, verify entities.
#[cfg(feature = "native")]
#[test]
fn dxf_import_parses_lines_and_circles() {
    use std::io::Write;
    use linkage_sim_rs::gui::dxf_import::{parse_dxf_file, DxfEntityKind};

    // Minimal DXF with 2 lines and 2 circles (simple 4-bar sketch in mm)
    let dxf_content = r#"0
SECTION
2
ENTITIES
0
LINE
8
0
10
0.0
20
0.0
11
100.0
21
0.0
0
LINE
8
0
10
100.0
20
0.0
11
100.0
21
50.0
0
CIRCLE
8
0
10
0.0
20
0.0
40
2.0
0
CIRCLE
8
0
10
100.0
20
50.0
40
2.0
0
ENDSEC
0
EOF
"#;

    let tmp = std::env::temp_dir().join("test_dxf_import.dxf");
    {
        let mut f = std::fs::File::create(&tmp).unwrap();
        f.write_all(dxf_content.as_bytes()).unwrap();
    }

    let overlay = parse_dxf_file(&tmp, 0.001).expect("parse failed");

    // Should have 4 entities (2 lines + 2 circles)
    assert_eq!(overlay.entities.len(), 4, "expected 4 entities");

    // Should have 2 snap circles
    assert_eq!(overlay.snap_circles.len(), 2, "expected 2 snap circles");

    // Check scale applied (mm -> m): first circle at origin, second at (0.1, 0.05)
    let c0 = &overlay.snap_circles[0];
    let c1 = &overlay.snap_circles[1];
    // Order may vary, find by position
    let at_origin = if c0.center[0].abs() < 1e-6 { c0 } else { c1 };
    let at_corner = if c0.center[0].abs() < 1e-6 { c1 } else { c0 };
    assert!((at_origin.center[0] - 0.0).abs() < 1e-6);
    assert!((at_origin.center[1] - 0.0).abs() < 1e-6);
    assert!((at_corner.center[0] - 0.1).abs() < 1e-6);
    assert!((at_corner.center[1] - 0.05).abs() < 1e-6);

    // Verify line entities
    let n_lines = overlay.entities.iter().filter(|e| matches!(e.kind, DxfEntityKind::Line { .. })).count();
    assert_eq!(n_lines, 2);

    // Verify snap works
    let snap = overlay.find_snap(0.0, 0.0, 0.01);
    assert!(snap.is_some(), "should snap to origin circle");
    let snap_pos = snap.unwrap();
    assert!((snap_pos[0]).abs() < 1e-6);
    assert!((snap_pos[1]).abs() < 1e-6);

    // Should not snap if too far away
    let no_snap = overlay.find_snap(1.0, 1.0, 0.01);
    assert!(no_snap.is_none(), "should not snap when far");

    std::fs::remove_file(&tmp).ok();
}
