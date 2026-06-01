use super::*;

// T-022: resolve_env_filter_accepts_multi_directive
#[test]
fn resolve_env_filter_accepts_multi_directive() {
    let _filter = resolve_env_filter("sae=info,hyper=warn");
}

// T-023: merge_default_directives_appends_upstream_warns
#[test]
fn merge_default_directives_appends_upstream_warns() {
    assert_eq!(
        merge_default_directives("yomu=warn"),
        "yomu=warn,rurico=warn,amici=warn"
    );
}

// T-024: merge_default_directives_handles_empty_default
#[test]
fn merge_default_directives_handles_empty_default() {
    assert_eq!(merge_default_directives(""), "rurico=warn,amici=warn");
}

// T-025: merge_default_directives_produces_parseable_filter
#[test]
fn merge_default_directives_produces_parseable_filter() {
    let merged = merge_default_directives("sae=info,hyper=warn");
    EnvFilter::try_new(&merged).expect("merged default filter must parse");
}
