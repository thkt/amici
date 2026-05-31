use super::*;

const KNOWN: &[&str] = &["harvest", "search"];
const GLOBAL: &[&str] = &["--json"];

fn os(s: &[&str]) -> Vec<OsString> {
    s.iter().map(|&a| a.into()).collect()
}

// T-008: 非サブコマンドの query → search を挿入
#[test]
fn bare_query_expands_to_search() {
    let exp = try_expand_shorthand(&os(&["sae", "認証"]), KNOWN, GLOBAL).unwrap();
    let s: Vec<&str> = exp.iter().filter_map(|a| a.to_str()).collect();
    assert_eq!(s, ["sae", "search", "認証"]);
}

// T-009: global flag は search の前に hoisted
#[test]
fn global_flag_hoisted_before_search() {
    let exp = try_expand_shorthand(&os(&["sae", "--json", "query"]), KNOWN, GLOBAL).unwrap();
    let s: Vec<&str> = exp.iter().filter_map(|a| a.to_str()).collect();
    assert_eq!(s, ["sae", "--json", "search", "query"]);
}

// T-010: known subcommand → None
#[test]
fn known_subcommand_not_expanded() {
    assert!(try_expand_shorthand(&os(&["sae", "harvest", "foo"]), KNOWN, GLOBAL).is_none());
}

// T-011: OSA distance=1 の typo → None（typo guard）
#[test]
fn typo_within_distance_not_expanded() {
    assert!(
        try_expand_shorthand(&os(&["sae", "serach"]), KNOWN, GLOBAL).is_none(),
        "typo 'serach' (osa=1 from 'search') should not expand"
    );
}

// T-023: global flag と trailing options が両立する（hoisting + 後続 flag 保持）
#[test]
fn global_flag_hoisted_with_trailing_options() {
    let exp = try_expand_shorthand(
        &os(&["sae", "--json", "query", "--limit", "2"]),
        KNOWN,
        GLOBAL,
    )
    .unwrap();
    let s: Vec<&str> = exp.iter().filter_map(|a| a.to_str()).collect();
    assert_eq!(s, ["sae", "--json", "search", "query", "--limit", "2"]);
}
