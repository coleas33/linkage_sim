pub mod error;
pub mod from_json;
pub mod schema;
pub mod to_json;

// Keep the old module around for its tests only.
#[cfg(test)]
mod serialization;

// Re-export for backward compatibility -- all public items from the old serialization module.
pub use error::SerializationError;
pub use from_json::{load_mechanism, load_mechanism_unbuilt, load_mechanism_unbuilt_from_json};
pub use schema::*;
pub use to_json::{mechanism_to_json, save_mechanism};
