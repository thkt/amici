//! FTS5 trigram tokenizer adapters for `rurico::storage::MatchFtsQuery` output.

use rurico::storage::MatchFtsQuery;

/// Upper bound on `cross_product` output size — above this, OR-groups are dropped.
///
/// Prevents the O(Π m_i) memory blowup that occurs when short tokens vocab-expand
/// to `(t1 OR t2 OR ... up to 25)` via `prepare_match_query` and stack with
/// multiple OR-groups (worst observed: `8 × 25 × 25 × 12 × 20 × 25 = 30M` combos,
/// ≈ 10 MB MATCH string, ≈ 81 GB resident — see issue #24).
///
/// At 100 the MATCH string stays under ~10 KB; downstream rerankers recover the
/// recall lost on the small fraction of queries that hit the fallback
/// (rurico eval: mrr / ndcg_at_10 unchanged versus a 10,000 cap).
const MAX_COMBOS: usize = 100;

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
///
/// When the OR-distribution would generate more than [`MAX_COMBOS`] flat
/// alternatives, the OR-groups are dropped and only `fixed` terms are used —
/// a recall-loss fallback that prevents pathological memory blowup.
pub fn clean_for_trigram(query: &MatchFtsQuery) -> Option<String> {
    clean_impl(query.as_str())
}

fn clean_impl(match_query: &str) -> Option<String> {
    let cleaned: String = match_query.chars().filter(|c| !c.is_control()).collect();
    let (fixed, or_groups) = parse_fts_segments(&cleaned);
    let fixed_only = || (!fixed.is_empty()).then(|| fixed.join(" "));

    if or_groups.is_empty() {
        return fixed_only();
    }

    // Estimate Π m_i without materializing cross_product. Saturates at
    // usize::MAX on overflow so the threshold check still fires.
    let estimated_combos = or_groups
        .iter()
        .map(Vec::len)
        .try_fold(1usize, usize::checked_mul)
        .unwrap_or(usize::MAX);
    if estimated_combos > MAX_COMBOS {
        return fixed_only();
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
mod tests;
