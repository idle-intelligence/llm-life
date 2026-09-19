//! Fine-tuning variant B (CONCEPT.md §5): the autodiff-capable forward,
//! LoRA adapters, the Life data generator and the training loop.
//!
//! Native only. The forward here is pure Burn tensor ops — no custom WGSL —
//! because autodiff cannot see through llm-web's hand-written kernels; the
//! price is f32 weights in memory (~2 GB for a 0.5B model) and the
//! obligation to prove the two agree, which `tests/train_oracle.rs` does.

pub mod data;
pub mod load;
pub mod lora_io;
pub mod model;
pub mod optim;
pub mod run;
pub mod weights;

pub use load::{load_train_model, LoraSpec};
pub use model::{cell_cross_entropy, Lora, TrainLayer, TrainLinear, TrainModel};
