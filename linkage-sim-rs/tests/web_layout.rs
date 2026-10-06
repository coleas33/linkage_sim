//! The web output's layout on colesorkness.com (plan 2026-10-06): the hub page
//! at `/`, the linkage app at `/linkage/`, the magnetic coupling calculator at
//! `/magcoupler/`, and the redirects that keep linkage.colesorkness.com links
//! working. The pages, `web/vercel.json`, `web/.gitignore` and the build
//! scripts each name these paths; these tests keep them in step.

use serde_json::Value;

const WEB: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/web");
const SCRIPTS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/scripts");

fn read(path: &str) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"))
}

fn vercel() -> Value {
    serde_json::from_str(&read(&format!("{WEB}/vercel.json"))).expect("vercel.json parses")
}

/// The app pages: (page, the glue it imports, its build script, the script's output folder).
const APPS: [(&str, &str, &str, &str); 2] = [
    ("linkage/index.html", "/linkage/linkage-web.js", "build_web.sh", "--out-dir web/linkage"),
    ("magcoupler/index.html", "/magcoupler/magcoupling-web.js", "build_magcoupling_web.sh", "OUT_DIR=\"$PROJECT_DIR/web/magcoupler\""),
];

#[test]
fn each_app_page_imports_its_glue_where_its_build_script_writes_it() {
    for (page, glue, script, out_dir) in APPS {
        let html = read(&format!("{WEB}/{page}"));
        assert!(html.contains(&format!("import init from '{glue}';")), "{page} imports {glue}");
        assert!(html.contains("href=\"/favicon.svg\""), "{page} uses the site's favicon");
        assert!(read(&format!("{SCRIPTS}/{script}")).contains(out_dir), "{script} writes to {out_dir}");
        // The generated glue and wasm are not committed.
        let ignored = read(&format!("{WEB}/.gitignore"));
        let glue_file = glue.trim_start_matches('/');
        let wasm_file = glue_file.replace(".js", "_bg.wasm");
        for file in [glue_file, wasm_file.as_str()] {
            assert!(ignored.lines().any(|line| line == file), ".gitignore lists {file}");
        }
    }
}

#[test]
fn the_hub_page_links_both_apps_and_loads_nothing_from_elsewhere() {
    let html = read(&format!("{WEB}/index.html"));
    for link in ["href=\"/linkage/\"", "href=\"/magcoupler/\""] {
        assert!(html.contains(link), "the hub links {link}");
    }
    assert!(!html.contains("http://") && !html.contains("https://"), "no resource from another site");
}

#[test]
fn the_glue_is_never_cached_and_wasm_has_its_type() {
    let v = vercel();
    let headers = v["headers"].as_array().expect("headers");
    for (_, glue, _, _) in APPS {
        assert!(headers.iter().any(|h| h["source"] == glue), "a cache rule for {glue}");
    }
    assert!(headers.iter().any(|h| h["source"] == "/(.*)\\.wasm"), "the wasm rule");
}

#[test]
fn folders_get_their_trailing_slash() {
    // Without it, /linkage would serve the page with the site root as its base.
    assert_eq!(vercel()["trailingSlash"], true);
}

/// The redirects, in order (Vercel applies the first that matches).
fn redirects() -> Vec<(String, String, Option<String>, Option<String>)> {
    vercel()["redirects"]
        .as_array()
        .expect("redirects")
        .iter()
        .map(|r| {
            assert_eq!(r["permanent"], true, "{r}");
            let has = r["has"].as_array().cloned().unwrap_or_default();
            let host = has.iter().find(|h| h["type"] == "host").map(|h| h["value"].as_str().unwrap().to_string());
            let query = has
                .iter()
                .find(|h| h["type"] == "query")
                .map(|h| format!("{}={}", h["key"].as_str().unwrap(), h["value"].as_str().unwrap()));
            (r["source"].as_str().unwrap().to_string(), r["destination"].as_str().unwrap().to_string(), host, query)
        })
        .collect()
}

#[test]
fn old_links_redirect_to_the_new_addresses() {
    let old = Some("linkage.colesorkness.com".to_string());
    assert_eq!(
        redirects(),
        [
            // The calculator's old page, share links included.
            ("/magcoupling/:path*".into(), "https://colesorkness.com/magcoupler/:path*".into(), old.clone(), None),
            // The linkage app with the embedded calculator open.
            ("/".into(), "https://colesorkness.com/magcoupler/".into(), old.clone(), Some("tool=magcoupling".into())),
            // Everything else on the old host, ?m= share links included, last.
            ("/:path*".into(), "https://colesorkness.com/linkage/:path*".into(), old, None),
            // The old calculator path on the new host.
            ("/magcoupling/:path*".into(), "/magcoupler/:path*".into(), None, None),
        ]
    );
}

#[test]
fn local_serving_checks_both_bundles_where_the_builds_put_them() {
    let serve = read(&format!("{SCRIPTS}/serve_web.sh"));
    for wasm in ["$WEB_DIR/linkage/linkage-web_bg.wasm", "$WEB_DIR/magcoupler/magcoupling-web_bg.wasm"] {
        assert!(serve.contains(wasm), "serve_web.sh checks {wasm}");
    }
}
