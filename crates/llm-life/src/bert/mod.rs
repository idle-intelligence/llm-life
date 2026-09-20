//! BERT of Life (CONCEPT.md §11): a small transformer encoder trained from
//! scratch on the same 512-case data as variant A, to show the size a model
//! needs to learn Life when the architecture is chosen for it instead of
//! borrowed from language.

pub mod data;
pub mod model;
#[cfg(feature = "native")]
pub mod train;
