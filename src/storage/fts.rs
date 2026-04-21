//! FTS5 trigram tokenizer adapters for `rurico::storage::MatchFtsQuery` output.

use rurico::storage::MatchFtsQuery;

/// Adapt a [`MatchFtsQuery`] for an FTS5 `trigram` tokenizer.
///
/// The trigram tokenizer rejects `("a" OR "b") "c"` (a parenthesized OR-group
/// followed by implicit AND). This function distributes OR-groups into flat
/// alternatives — `(A OR B) C` → `A C OR B C` — and drops sub-trigram terms
/// (<3 chars) inside OR-groups because the trigram tokenizer cannot index them.
///
/// Control characters are stripped. Returns `None` when the input contains no
/// indexable terms (e.g. every OR-group alternative was sub-trigram); callers
/// should treat this as "no results" rather than passing an empty string to
/// FTS5 `MATCH`.
pub fn clean_for_trigram(query: &MatchFtsQuery) -> Option<String> {
    clean_impl(query.as_str())
}

fn clean_impl(match_query: &str) -> Option<String> {
    let cleaned: String = match_query.chars().filter(|c| !c.is_control()).collect();
    let (fixed, or_groups) = parse_fts_segments(&cleaned);

    if or_groups.is_empty() {
        if fixed.is_empty() {
            return None;
        }
        return Some(fixed.join(" "));
    }

    let combos = cross_product(&or_groups);
    Some(
        combos
            .iter()
            .map(|combo| {
                let mut parts = combo.clone();
                parts.extend(fixed.iter().cloned());
                parts.join(" ")
            })
            .collect::<Vec<_>>()
            .join(" OR "),
    )
}

fn parse_fts_segments(cleaned: &str) -> (Vec<String>, Vec<Vec<String>>) {
    let mut fixed: Vec<String> = Vec::new();
    let mut or_groups: Vec<Vec<String>> = Vec::new();
    let mut chars = cleaned.chars();

    while let Some(c) = chars.next() {
        if c == '(' {
            let mut group = String::new();
            for gc in chars.by_ref() {
                if gc == ')' {
                    break;
                }
                group.push(gc);
            }
            let terms: Vec<String> = group
                .split(" OR ")
                .filter(|t| t.trim().trim_matches('"').chars().count() >= 3)
                .map(|t| t.trim().to_owned())
                .collect();
            if !terms.is_empty() {
                or_groups.push(terms);
            }
        } else if c == '"' {
            let mut term = String::from('"');
            for tc in chars.by_ref() {
                term.push(tc);
                if tc == '"' {
                    break;
                }
            }
            // Trigram tokenizer cannot index <3 char terms; apply the same
            // filter used inside OR-groups so the Option<String> contract
            // stays honest across all input shapes.
            if term.trim_matches('"').chars().count() >= 3 {
                fixed.push(term);
            }
        }
    }

    (fixed, or_groups)
}

fn cross_product(groups: &[Vec<String>]) -> Vec<Vec<String>> {
    if groups.is_empty() {
        return vec![vec![]];
    }
    let rest = cross_product(&groups[1..]);
    let mut result = Vec::new();
    for term in &groups[0] {
        for combo in &rest {
            let mut v = vec![term.clone()];
            v.extend(combo.iter().cloned());
            result.push(v);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use rurico::storage::prepare_match_query;
    use rusqlite::Connection;

    #[test]
    fn distributes_or_groups() {
        // Control chars removed + sub-trigram dropped + distributed
        assert_eq!(
            clean_impl("(\"認証の\" OR \"認証\n\" OR \"認証フ\") \"フロー\"").as_deref(),
            Some("\"認証の\" \"フロー\" OR \"認証フ\" \"フロー\"")
        );

        // Multi-element group + fixed term → distributed
        assert_eq!(
            clean_impl("(\"abc\" OR \"def\") \"ghi\"").as_deref(),
            Some("\"abc\" \"ghi\" OR \"def\" \"ghi\"")
        );

        // No parens → unchanged
        assert_eq!(clean_impl("\"hello\"").as_deref(), Some("\"hello\""));

        // Single group, no fixed terms → just OR
        assert_eq!(
            clean_impl("(\"abc\" OR \"def\")").as_deref(),
            Some("\"abc\" OR \"def\"")
        );

        // Multiple OR groups → cross-product
        assert_eq!(
            clean_impl("(\"a01\" OR \"a02\") (\"b01\" OR \"b02\")").as_deref(),
            Some("\"a01\" \"b01\" OR \"a01\" \"b02\" OR \"a02\" \"b01\" OR \"a02\" \"b02\"")
        );
    }

    #[test]
    fn returns_none_when_all_sub_trigram() {
        // All terms in the only OR-group are sub-trigram (<3 chars).
        // parse_fts_segments filters them, leaving no fixed terms → None.
        assert_eq!(clean_impl("(\"ab\" OR \"cd\")"), None);
    }

    #[test]
    fn returns_none_for_empty_input() {
        assert_eq!(clean_impl(""), None);
    }

    #[test]
    fn returns_none_for_sub_trigram_fixed_terms() {
        // All fixed quoted tokens are <3 chars — trigram tokenizer cannot index them.
        assert_eq!(clean_impl("\"au\""), None);
        assert_eq!(clean_impl("\"a\" \"b\""), None);
    }

    #[test]
    fn accepts_live_prepare_match_query_output() {
        // Integration: rurico sanitizes, amici adapts and returns Some.
        let conn = Connection::open_in_memory().unwrap();
        let matched = prepare_match_query(&conn, "hello world", "nonexistent_vocab").unwrap();
        assert_eq!(
            clean_for_trigram(&matched).as_deref(),
            Some("\"hello\" \"world\"")
        );
    }
}
