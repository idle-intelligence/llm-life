//! A language model as the update rule of a cellular automaton.
//!
//! The classical rule lives in the `life` crate and is the ground truth; this
//! crate owns the CA-to-model glue — packing, stencil masks, reading p(alive)
//! out of logits — and scores the model against that ground truth.

pub mod pgm;
pub mod score;
pub mod variant_b;

#[cfg(feature = "web")]
pub mod web;
