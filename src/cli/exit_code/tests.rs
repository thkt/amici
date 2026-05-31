use super::codes;

// sysexits.h reference values pinned here so a future refactor that
// touches `codes::*` fails visibly instead of silently breaking
// downstream CLIs that depend on these numbers. Project-extension-range
// codes (UNKNOWN, INTERNAL) live in a separate test so a reader can tell
// sysexits-derived codes from project extensions by which test owns them.

// T-122: codes_match_sysexits_constants
#[test]
fn codes_match_sysexits_constants() {
    assert_eq!(codes::SUCCESS, 0);
    assert_eq!(codes::USAGE, 64);
    assert_eq!(codes::DATA_ERROR, 65);
    assert_eq!(codes::SOFTWARE, 70);
    assert_eq!(codes::CANT_CREAT, 73);
    assert_eq!(codes::IO_ERR, 74);
    assert_eq!(codes::TEMP_FAIL, 75);
}

// T-123: codes_match_pj_extension_constants
#[test]
fn codes_match_pj_extension_constants() {
    // PJ extension range (80–119), per ADR-0066.
    assert_eq!(codes::UNKNOWN, 104);

    // INTERNAL is an alias for SOFTWARE; pin both that it resolves to
    // the same constant and that the numeric value is EX_SOFTWARE (70).
    assert_eq!(codes::INTERNAL, codes::SOFTWARE);
    assert_eq!(codes::INTERNAL, 70);
}
