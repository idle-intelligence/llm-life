//! A language model as the update rule of a cellular automaton.
//!
//! The classical rule lives in the `life` crate and is the ground truth; this
//! crate owns the CA-to-model glue — packing, stencil masks, reading p(alive)
//! out of logits — and scores the model against that ground truth.

// burn's `#[derive(Config)]` generates a `new(...)` constructor with
// field-init-shorthand-shaped `Self { field: field, ... }` assignments;
// clippy attributes redundant_field_names to the field declaration, not
// anything this crate's own code wrote. Silenced crate-wide rather than
// per struct, since every `*Config` type triggers it the same way.
#![allow(clippy::redundant_field_names)]

pub mod bert;
pub mod pgm;
pub mod score;
#[cfg(feature = "native")]
pub mod train;
pub mod variant_a;
pub mod variant_b;
pub mod vector;

#[cfg(feature = "web")]
pub mod web;
