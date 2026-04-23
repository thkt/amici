//! FTS↔vec symmetry contract tests for hybrid search.
//!
//! Hybrid search (FTS + vec) routes one query through two paths and merges
//! the results. A recurring bug class is "filter applied on FTS path but
//! ignored on vec path", which lets filter-violating items leak through.
//!
//! - recall #35: `session_filter` ignored on vec path — fixed with a
//!   post-filter `retain` at the merge layer.
//! - yomu #103: `type_filter` ignored on vec path — fixed by appending the
//!   filter to the vec SQL directly.
//!
//! Two different fix strategies, same contract. The helper in this module
//! treats the search as a black box and asserts the outcome, so it works
//! regardless of where the caller decided to enforce the filter.

use std::fmt::Debug;

/// Assert that a hybrid search applies a filter on both FTS and vec paths.
///
/// The caller supplies three closures:
///
/// 1. `setup` — seed the store and return `(should_appear, should_not_appear)`.
///    - `should_appear`: passes FTS AND satisfies the filter.
///    - `should_not_appear`: matches vec only (no FTS hit) AND violates the
///      filter.
/// 2. `search_filtered` — run hybrid search with the filter enabled; return
///    the IDs in the results.
/// 3. `search_unfiltered` — run the same hybrid search with the filter
///    disabled. This is a probe: `should_not_appear` must surface here,
///    otherwise the test setup is broken (the seed is unreachable on the vec
///    path, or FTS is secretly indexing it).
///
/// # Panics
///
/// Panics if any of the following does not hold:
///
/// - probe: `should_not_appear` is absent from `search_unfiltered()`.
///   Diagnoses a broken setup before blaming the implementation.
/// - positive: `should_appear` is absent from `search_filtered()`.
/// - contract: `should_not_appear` leaks into `search_filtered()` — the
///   actual FTS↔vec asymmetry bug.
///
/// # Examples
///
/// ```ignore
/// use amici::testing::hybrid::assert_filter_symmetric;
///
/// assert_filter_symmetric(
///     || {
///         // seed_a: FTS-hitting, filter-passing
///         insert_chunk(&conn, 1, "authentication logic", ChunkType::Function);
///         // seed_b: vec-only, filter-violating
///         insert_chunk(&conn, 2, "authenticate", ChunkType::Test);
///         (1_i64, 2_i64)
///     },
///     || search(&conn, "auth", Some(&[ChunkType::Function]))
///         .into_iter().filter_map(|r| r.chunk_id).collect(),
///     || search(&conn, "auth", None)
///         .into_iter().filter_map(|r| r.chunk_id).collect(),
/// );
/// ```
pub fn assert_filter_symmetric<Id>(
    setup: impl FnOnce() -> (Id, Id),
    search_filtered: impl FnOnce() -> Vec<Id>,
    search_unfiltered: impl FnOnce() -> Vec<Id>,
) where
    Id: PartialEq + Debug,
{
    let (should_appear, should_not_appear) = setup();

    let unfiltered = search_unfiltered();
    assert!(
        unfiltered.contains(&should_not_appear),
        "probe failed: {should_not_appear:?} was not returned by the unfiltered search. \
         The seed must match on the vec path for this contract to be meaningful; \
         check the seed's content/embedding before blaming the implementation. \
         unfiltered results: {unfiltered:?}",
    );

    let filtered = search_filtered();
    assert!(
        filtered.contains(&should_appear),
        "expected {should_appear:?} (FTS-hitting, filter-passing seed) in filtered results, \
         got: {filtered:?}",
    );
    assert!(
        !filtered.contains(&should_not_appear),
        "FTS↔vec asymmetry: {should_not_appear:?} (vec-only match, filter-violating) \
         leaked into filtered results {filtered:?}. \
         The filter is honored on the FTS path but ignored on the vec path — \
         apply it on both, or post-filter the merged output.",
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    // T-025: asymmetry_bug_panics_with_diagnostic
    #[test]
    #[should_panic(expected = "FTS↔vec asymmetry")]
    fn asymmetry_bug_panics_with_diagnostic() {
        // Buggy impl: filter is not applied on the vec path, so "b" leaks in.
        assert_filter_symmetric(|| ("a", "b"), || vec!["a", "b"], || vec!["a", "b"]);
    }

    // T-026: missing_should_appear_panics
    #[test]
    #[should_panic(expected = "filter-passing seed")]
    fn missing_should_appear_panics() {
        // Impl returns nothing for the filtered search — "a" disappears too.
        assert_filter_symmetric(|| ("a", "b"), Vec::new, || vec!["a", "b"]);
    }

    // T-027: vacuous_pass_caught_by_probe
    #[test]
    #[should_panic(expected = "probe failed")]
    fn vacuous_pass_caught_by_probe() {
        // Seed "b" is not actually findable on the vec path — broken setup.
        // Without the probe, the filtered assertion would pass vacuously.
        assert_filter_symmetric(|| ("a", "b"), || vec!["a"], || vec!["a"]);
    }

    // T-028: correct_impl_passes
    #[test]
    fn correct_impl_passes() {
        // Correct impl: "b" matches vec without filter but is filtered out.
        assert_filter_symmetric(|| ("a", "b"), || vec!["a"], || vec!["a", "b"]);
    }

    // T-029: supports_numeric_ids
    #[test]
    fn supports_numeric_ids() {
        // Verify the Id generic parameter accepts non-string types (i64, newtype).
        assert_filter_symmetric::<i64>(|| (1, 2), || vec![1], || vec![1, 2]);
    }
}
