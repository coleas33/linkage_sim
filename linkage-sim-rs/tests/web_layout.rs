//! The web output's layout on colesorkness.com (plan 2026-10-06): the root
//! redirects to the hub page at `/tools/`, the linkage app is at
//! `/tools/linkage/`, the magnetic coupling calculator at `/tools/magcoupler/`,
//! and redirects keep every linkage.colesorkness.com link working. The pages,
//! `web/vercel.json`, `web/.gitignore` and the build scripts each name these
//! paths; these tests keep them in step.

use serde_json::{json, Value};

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
    ("tools/linkage/index.html", "/tools/linkage/linkage-web.js", "build_web.sh", "--out-dir web/tools/linkage"),
    (
        "tools/magcoupler/index.html",
        "/tools/magcoupler/magcoupling-web.js",
        "build_magcoupling_web.sh",
        "OUT_DIR=\"$PROJECT_DIR/web/tools/magcoupler\"",
    ),
];

#[test]
fn each_app_page_imports_its_glue_where_its_build_script_writes_it() {
    for (page, glue, script, out_dir) in APPS {
        let html = read(&format!("{WEB}/{page}"));
        assert!(html.contains(&format!("import init from '{glue}';")), "{page} imports {glue}");
        assert!(html.contains("href=\"/favicon.svg\""), "{page} uses the site's favicon");
        assert!(read(&format!("{SCRIPTS}/{script}")).contains(out_dir), "{script} writes to {out_dir}");
        // The generated glue and wasm are not committed: the patterns match at any depth.
        let ignored = read(&format!("{WEB}/.gitignore"));
        let glue_file = glue.rsplit('/').next().unwrap();
        let wasm_file = glue_file.replace(".js", "_bg.wasm");
        for file in [glue_file, wasm_file.as_str()] {
            assert!(ignored.lines().any(|line| line == file), ".gitignore lists {file}");
        }
    }
}

#[test]
fn the_hub_page_links_both_apps_and_loads_nothing_from_elsewhere() {
    let html = read(&format!("{WEB}/tools/index.html"));
    for link in ["href=\"/tools/linkage/\"", "href=\"/tools/magcoupler/\""] {
        assert!(html.contains(link), "the hub links {link}");
    }
    for elsewhere in ["http://", "https://", "\"//", "'//"] {
        assert!(!html.contains(elsewhere), "no resource from another site ({elsewhere})");
    }
}

#[test]
fn the_root_page_forwards_to_the_hub_for_a_server_without_the_redirect() {
    let html = read(&format!("{WEB}/index.html"));
    assert!(html.contains("<meta http-equiv=\"refresh\" content=\"0; url=/tools/\">"));
    assert!(html.contains("href=\"/tools/\""));
}

#[test]
fn the_glue_is_never_cached_and_wasm_has_its_type() {
    let revalidate = json!({ "key": "Cache-Control", "value": "public, max-age=0, must-revalidate" });
    let headers = vercel()["headers"].clone();
    let expected = json!([
        { "source": "/(.*)\\.wasm", "headers": [{ "key": "Content-Type", "value": "application/wasm" }, revalidate] },
        { "source": APPS[0].1, "headers": [revalidate] },
        { "source": APPS[1].1, "headers": [revalidate] },
    ]);
    assert_eq!(headers, expected);
}

#[test]
fn folders_get_their_trailing_slash() {
    // Without it, /tools/linkage would serve the page with /tools/ as its base.
    assert_eq!(vercel()["trailingSlash"], true);
}

#[test]
fn old_links_and_the_root_redirect_to_the_new_addresses() {
    let old = json!({ "type": "host", "value": "linkage.colesorkness.com" });
    let m = json!({ "type": "query", "key": "m", "value": "(?<m>.*)" });
    // The old addresses redirect permanently (308, decision 4A); the root's stays temporary.
    let expected = json!([
        // The calculator's old page, its share links (?m=) kept explicitly.
        { "source": "/magcoupling/:path(.*)", "has": [old, m],
          "destination": "https://colesorkness.com/tools/magcoupler/:path?m=:m", "permanent": true },
        { "source": "/magcoupling/:path(.*)", "has": [old],
          "destination": "https://colesorkness.com/tools/magcoupler/:path", "permanent": true },
        // The linkage app with the embedded calculator open (a mechanism link wins).
        { "source": "/", "has": [old, { "type": "query", "key": "tool", "value": "magcoupling" }],
          "missing": [{ "type": "query", "key": "m" }],
          "destination": "https://colesorkness.com/tools/magcoupler/", "permanent": true },
        // Everything else on the old host, ?m= share links kept explicitly.
        { "source": "/:path(.*)", "has": [old, m],
          "destination": "https://colesorkness.com/tools/linkage/:path?m=:m", "permanent": true },
        { "source": "/:path(.*)", "has": [old],
          "destination": "https://colesorkness.com/tools/linkage/:path", "permanent": true },
        // The old calculator path on the new host.
        { "source": "/magcoupling/:path(.*)", "has": [m],
          "destination": "/tools/magcoupler/:path?m=:m", "permanent": true },
        { "source": "/magcoupling/:path(.*)", "destination": "/tools/magcoupler/:path", "permanent": true },
        // The root, to the hub (temporary: a page of its own may replace it).
        { "source": "/", "destination": "/tools/", "permanent": false },
    ]);
    assert_eq!(vercel()["redirects"], expected);
}

/// Whether a redirect source has a named parameter (`:name`) directly followed by `*`.
fn has_star_parameter(source: &str) -> bool {
    source.split(':').skip(1).any(|rest| {
        let name_len = rest.chars().take_while(|c| c.is_ascii_alphanumeric() || *c == '_').count();
        name_len > 0 && rest[name_len..].starts_with('*')
    })
}

#[test]
fn no_redirect_uses_a_star_parameter() {
    // Vercel compiles `:path*` strictly: `^/magcoupling(?:/((?:[^/]+?)(?:/(?:[^/]+?))*))?$` never
    // matches `/magcoupling/`, the form of every share link the calculator wrote. `:path(.*)`
    // matches it.
    for r in vercel()["redirects"].as_array().unwrap() {
        let source = r["source"].as_str().unwrap();
        assert!(!has_star_parameter(source), "{source}");
    }
    assert!(has_star_parameter("/magcoupling/:path*") && !has_star_parameter("/magcoupling/:path(.*)"));
}

#[test]
fn local_serving_checks_both_bundles_where_the_builds_put_them() {
    let serve = read(&format!("{SCRIPTS}/serve_web.sh"));
    for wasm in ["$WEB_DIR/tools/linkage/linkage-web_bg.wasm", "$WEB_DIR/tools/magcoupler/magcoupling-web_bg.wasm"] {
        assert!(serve.contains(wasm), "serve_web.sh checks {wasm}");
    }
}
