//! Classical cellular automaton: the ground truth the LLM is scored against.
//!
//! Deliberately dependency-free and identical on native and wasm32 — the
//! `web` feature only adds a wasm-bindgen wrapper (`app`), it never changes
//! the rule.

pub mod grid;
pub mod rule;

#[cfg(feature = "web")]
pub mod app;

pub use grid::Grid;
pub use rule::Rule;
