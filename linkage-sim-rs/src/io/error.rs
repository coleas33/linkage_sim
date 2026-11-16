//! Serialization error types.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum SerializationError {
    #[error("Unsupported schema version '{found}' (expected major version compatible with '{expected}').")]
    UnsupportedVersion { found: String, expected: String },

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Failed to build mechanism: {0}")]
    Build(String),

    #[error(
        "Could not find attachment point name on body '{body_id}' \
         for local coordinates [{x}, {y}]"
    )]
    PointNameNotFound {
        body_id: String,
        x: f64,
        y: f64,
    },
}

impl SerializationError {
    /// A [`SerializationError::Build`] carrying `cause`'s message. Used as
    /// `.map_err(SerializationError::build)` on any builder error.
    pub(crate) fn build(cause: impl std::fmt::Display) -> Self {
        SerializationError::Build(cause.to_string())
    }
}
