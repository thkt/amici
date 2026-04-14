use std::ffi::OsString;

/// Expands shorthand `<bin> "query"` → `<bin> [global_flags] search "query" [rest_flags]`.
///
/// Returns `Some(expanded_args)` when the first positional argument is not a known
/// subcommand and has OSA distance > 1 from all known subcommands, `None` otherwise.
pub fn try_expand_shorthand(
    args: &[OsString],
    known_subcommands: &[&str],
    global_flags: &[&str],
) -> Option<Vec<OsString>> {
    let positional_count = args
        .iter()
        .filter(|a| !a.to_str().is_some_and(|s| s.starts_with('-')))
        .count();

    if positional_count < 2 {
        return None;
    }

    let (flags, rest): (Vec<_>, Vec<_>) = args
        .iter()
        .enumerate()
        .partition(|(i, a)| *i > 0 && a.to_str().is_some_and(|s| global_flags.contains(&s)));
    let rest: Vec<&OsString> = rest.into_iter().map(|(_, a)| a).collect();

    if rest.len() >= 2
        && let Some(first_arg) = rest[1].to_str()
        && !first_arg.starts_with('-')
        && first_arg != "help"
        && !known_subcommands.contains(&first_arg)
        && !known_subcommands
            .iter()
            .any(|k| osa_distance(first_arg, k) <= 1)
    {
        let mut expanded: Vec<OsString> = vec![rest[0].clone()];
        for (_, f) in &flags {
            expanded.push((*f).clone());
        }
        expanded.push("search".into());
        for arg in &rest[1..] {
            expanded.push((*arg).clone());
        }
        Some(expanded)
    } else {
        None
    }
}

fn osa_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let (na, nb) = (a.len(), b.len());
    let mut d = vec![vec![0usize; nb + 1]; na + 1];
    for (i, row) in d.iter_mut().enumerate().take(na + 1) {
        row[0] = i;
    }
    for (j, cell) in d[0].iter_mut().enumerate().take(nb + 1) {
        *cell = j;
    }
    for i in 1..=na {
        for j in 1..=nb {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            d[i][j] = (d[i - 1][j] + 1)
                .min(d[i][j - 1] + 1)
                .min(d[i - 1][j - 1] + cost);
            if i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] {
                d[i][j] = d[i][j].min(d[i - 2][j - 2] + cost);
            }
        }
    }
    d[na][nb]
}

#[cfg(test)]
mod tests {
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
}
