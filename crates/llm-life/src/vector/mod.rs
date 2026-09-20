//! Vector-space variants of "learn Life from data" (CONCEPT.md §12): no
//! tokens anywhere. Two rungs:
//!
//! - `model::AttnOfLife` / the MLP baseline (reused from `bert::model`):
//!   the 3x3 neighbourhood as 9 floats -> the centre cell, same 512-case
//!   data as BERT of Life but with no embedding table.
//! - `model::StencilOfLife`: the whole grid, one channel, in one pass —
//!   jacobi2000's stencil-masked attention (`jacobi2000::mask::Mask`,
//!   `jacobi2000::model::Block`, read-only reference, not a dependency)
//!   with a single scalar channel and a sigmoid/BCE head instead of a
//!   multi-channel PDE state.

pub mod data;
pub mod model;
#[cfg(feature = "native")]
pub mod train;
