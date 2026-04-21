//! FTS5 trigram tokenizer adapters for `rurico::storage::MatchFtsQuery` output.

/// Adapt a [`MatchFtsQuery`](rurico::storage::MatchFtsQuery) string for an FTS5
/// `trigram` tokenizer.
///
/// The trigram tokenizer rejects `("a" OR "b") "c"` (a parenthesized OR-group
/// followed by implicit AND). This function distributes OR-groups into flat
/// alternatives — `(A OR B) C` → `A C OR B C` — and drops sub-trigram terms
/// (<3 chars) inside OR-groups because the trigram tokenizer cannot index them.
///
/// Callers should pass `rurico::storage::MatchFtsQuery::as_str()`; other
/// inputs are accepted for testability but bare tokens outside `(...)` or
/// `"..."` are silently dropped. Control characters are stripped.
///
/// Returns an empty string when every OR-group alternative was sub-trigram and
/// no fixed terms remain — callers should treat this as "no results".
pub fn clean_for_trigram(match_query: &str) -> String {
    let cleaned: String = match_query.chars().filter(|c| !c.is_control()).collect();
    let (fixed, or_groups) = parse_fts_segments(&cleaned);

    if or_groups.is_empty() {
        return fixed.join(" ");
    }

    let combos = cross_product(&or_groups);
    combos
        .iter()
        .map(|combo| {
            let mut parts = combo.clone();
            parts.extend(fixed.iter().cloned());
            parts.join(" ")
        })
        .collect::<Vec<_>>()
        .join(" OR ")
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
            fixed.push(term);
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
            clean_for_trigram("(\"認証の\" OR \"認証\n\" OR \"認証フ\") \"フロー\""),
            "\"認証の\" \"フロー\" OR \"認証フ\" \"フロー\""
        );

        // Multi-element group + fixed term → distributed
        assert_eq!(
            clean_for_trigram("(\"abc\" OR \"def\") \"ghi\""),
            "\"abc\" \"ghi\" OR \"def\" \"ghi\""
        );

        // No parens → unchanged
        assert_eq!(clean_for_trigram("\"hello\""), "\"hello\"");

        // Single group, no fixed terms → just OR
        assert_eq!(
            clean_for_trigram("(\"abc\" OR \"def\")"),
            "\"abc\" OR \"def\""
        );

        // Multiple OR groups → cross-product
        assert_eq!(
            clean_for_trigram("(\"a01\" OR \"a02\") (\"b01\" OR \"b02\")"),
            "\"a01\" \"b01\" OR \"a01\" \"b02\" OR \"a02\" \"b01\" OR \"a02\" \"b02\""
        );
    }

    #[test]
    fn empties_all_sub_trigram() {
        // All terms in the only OR-group are sub-trigram (<3 chars).
        // parse_fts_segments filters them, leaving no fixed terms → empty output.
        assert_eq!(clean_for_trigram("(\"ab\" OR \"cd\")"), "");
    }

    #[test]
    fn accepts_live_prepare_match_query_output() {
        // Integration sanity: clean_for_trigram accepts rurico output as-is.
        let conn = Connection::open_in_memory().unwrap();
        let matched = prepare_match_query(&conn, "hello world", "nonexistent_vocab").unwrap();
        let cleaned = clean_for_trigram(matched.as_str());
        assert_eq!(cleaned, "\"hello\" \"world\"");
    }
}
