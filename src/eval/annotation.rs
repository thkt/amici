//! Annotation framework foundation (Issue #53 Phase 1 sub-PR-A).
//!
//! Public types underpinning the offline annotation authoring tool whose
//! subcommand wiring lands in sub-PR-B (after #61 / #62). Phase 1
//! deliberately omits `model_id` and `mlx_rs_version` from
//! [`Provenance`] — those attach in Phase 1.5 Collaborative mode and
//! trigger an [`ANNOTATION_SCHEMA_VERSION`] bump per ADR-0004.
//!
//! The envelope is annotation-tailored (does not mirror
//! [`crate::eval::baseline::BaselineSnapshot`]) so authoring metadata
//! evolves independently of capture metadata. Migration triggers and
//! intentionally-omitted fields are documented in
//! `docs/decisions/0004-annotation-framework.md`.

use std::collections::BTreeMap;
use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};

/// Canonical schema-version label stamped onto every emitted
/// [`Provenance`] envelope. Bump when the envelope shape changes.
///
/// Version log:
/// - `1.0`: initial Phase 1 envelope. Block-mode authoring only;
///   `model_id` and `mlx_rs_version` are intentionally absent — see
///   `docs/decisions/0004-annotation-framework.md` §Decision Outcome
///   for the migration trigger that bumps to `1.1`.
pub const ANNOTATION_SCHEMA_VERSION: &str = "1.0";

/// Authoring strategy discriminator. Exactly one Phase 1 variant
/// (`Standard`) per FR-006; `Highlight` and `Collaborative` modes attach
/// in Phase 1.5+ and trigger an [`ANNOTATION_SCHEMA_VERSION`] bump.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlockMode {
    /// Standard block-mode authoring: relevance grades are filled in
    /// without model assistance.
    Standard,
}

/// Per-session authoring metadata captured alongside annotation
/// records. Phase 1 omits `model_id` / `mlx_rs_version` per ADR-0004.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    /// Envelope schema label. See [`ANNOTATION_SCHEMA_VERSION`].
    pub schema_version: String,
    /// Subcommand or context label that produced this session.
    pub captured_with: String,
    /// Capture-time label in `epoch:N` form. Avoids pulling `chrono`
    /// in just for an ISO-8601 timestamp; mirrors
    /// [`crate::eval::baseline::BaselineSnapshot::timestamp`].
    pub timestamp: String,
    /// Annotator identity (e.g. `thkt`, GitHub handle, OAuth subject).
    pub annotator_id: String,
    /// Stable session identifier used to deduplicate / correlate
    /// records across runs.
    pub session_id: String,
    /// Content hash over `queries.jsonl` at session start. Mirrors
    /// `BaselineSnapshot.fixture_hash` so authoring can be replayed
    /// against the exact fixture state it was authored against.
    pub queries_jsonl_hash: String,
}

/// Single annotation record. Carries graded relevance per `doc_id`
/// plus a free-form rationale note.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Entry {
    /// Stable identifier (typically mirrors a query / record id).
    pub id: String,
    /// Surface text being annotated.
    pub text: String,
    /// One of the seven semantic category labels (FR-006 of ADR-0002).
    pub category: String,
    /// Maps `doc_id` to graded relevance in `{0, 1, 2, 3}`. `BTreeMap`
    /// guarantees deterministic JSON key order so an unchanged
    /// [`Session`] re-emits byte-identical output across runs.
    pub relevance_map: BTreeMap<String, u8>,
    /// Free-form rationale note for the relevance judgment. Distinct
    /// from [`crate::eval::fixture::EvalQuery::annotation`] to avoid
    /// identifier collision in shared `use` scope (FR-005).
    pub annotation_note: String,
    /// Authoring strategy used (FR-006).
    pub mode: BlockMode,
}

/// One annotation session. Pairs [`Provenance`] with the ordered list of
/// records authored under it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Session {
    /// Capture metadata for this session.
    pub provenance: Provenance,
    /// Authored records in capture order.
    pub entries: Vec<Entry>,
}

impl Session {
    /// Verify [`Provenance::schema_version`] equals
    /// [`ANNOTATION_SCHEMA_VERSION`].
    ///
    /// # Errors
    ///
    /// Returns [`AnnotationError::SchemaVersionMismatch`] when the
    /// labels disagree, carrying both observed and expected labels so
    /// downstream consumers can render an actionable diff.
    pub fn validate_schema_version(&self) -> Result<(), AnnotationError> {
        if self.provenance.schema_version != ANNOTATION_SCHEMA_VERSION {
            return Err(AnnotationError::SchemaVersionMismatch {
                got: self.provenance.schema_version.clone(),
                expected: ANNOTATION_SCHEMA_VERSION.to_owned(),
            });
        }
        Ok(())
    }

    /// Atomic-write this session as pretty JSON to `path`.
    ///
    /// Delegates to [`super::io::write_json`] so [`Session`] and
    /// [`super::baseline::BaselineSnapshot`] share one atomic-write
    /// code path. `serde_json` failures are funneled through
    /// [`io::Error::other`] inside that helper, so callers only need
    /// to match [`AnnotationError::Io`] for both serialisation and
    /// filesystem faults.
    ///
    /// # Errors
    ///
    /// Returns [`AnnotationError::Io`] when the temp file cannot be
    /// created, written, fsynced, or renamed into place.
    pub fn write_json(&self, path: &Path) -> Result<(), AnnotationError> {
        super::io::write_json(self, path)?;
        Ok(())
    }
}

/// Errors surfaced by annotation framework operations.
#[derive(thiserror::Error, Debug)]
#[non_exhaustive]
pub enum AnnotationError {
    /// `Provenance.schema_version` did not equal
    /// [`ANNOTATION_SCHEMA_VERSION`] at validation time.
    #[error("annotation schema version mismatch (got {got:?}, expected {expected:?})")]
    SchemaVersionMismatch {
        /// Observed label carried by the provenance under inspection.
        got: String,
        /// Canonical label expected at validation.
        expected: String,
    },
    /// Annotation session contains zero entries.
    #[error("annotation session is empty")]
    EmptySession,
    /// JSON serialisation / deserialisation failure. Retained for
    /// forward compatibility — Phase 1 onwards, `serde_json::Error`
    /// raised inside [`Session::write_json`] is funneled via
    /// [`io::Error::other`] in [`super::io::write_json`] and surfaces as
    /// [`AnnotationError::Io`]. Future deserialise paths (e.g.
    /// `Session::read_json`) will fire this variant directly.
    #[error("annotation serialise error: {0}")]
    Serialise(#[from] serde_json::Error),
    /// I/O failure during atomic write or read of session JSON. Wraps
    /// both raw [`io::Error`] and `serde_json` failures funneled via
    /// [`io::Error::other`] in [`super::io::write_json`].
    #[error("annotation io error: {0}")]
    Io(#[from] io::Error),
}

#[cfg(test)]
mod tests;
