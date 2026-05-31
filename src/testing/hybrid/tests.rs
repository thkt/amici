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
