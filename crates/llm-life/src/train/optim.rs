//! AdamW over a flat list of LoRA matrices.
//!
//! Hand-rolled rather than `burn::optim::AdamW`: that optimizer drives a
//! `Module`'s `Param`s, and `TrainModel` is deliberately not a `Module` — its
//! base weights are frozen plain tensors and only the LoRA matrices move, so
//! "a `Vec<Tensor>` and its two moment buffers" is the whole state. Decoupled
//! weight decay (Loshchilov & Hutter 2019, arXiv:1711.05101), applied to the
//! parameter rather than to the gradient.

use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Tensor;

pub struct AdamW<B: AutodiffBackend> {
    m: Vec<Tensor<B::InnerBackend, 2>>,
    v: Vec<Tensor<B::InnerBackend, 2>>,
    t: i32,
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
}

impl<B: AutodiffBackend> AdamW<B> {
    pub fn new(params: &[Tensor<B, 2>], lr: f64) -> Self {
        let zeros = |t: &Tensor<B, 2>| Tensor::zeros(t.dims(), &t.device());
        Self {
            m: params.iter().map(zeros).collect(),
            v: params.iter().map(zeros).collect(),
            t: 0,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        }
    }

    /// One step. Returns the updated parameters, already `require_grad`-ed
    /// for the next forward.
    pub fn step(
        &mut self,
        params: Vec<Tensor<B, 2>>,
        grads: &B::Gradients,
    ) -> Vec<Tensor<B, 2>> {
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t);
        let bc2 = 1.0 - self.beta2.powi(self.t);
        let mut out = Vec::with_capacity(params.len());
        for (i, p) in params.into_iter().enumerate() {
            let g = p
                .grad(grads)
                .expect("every trainable parameter must have a gradient");
            let m = self.m[i].clone().mul_scalar(self.beta1) + g.clone().mul_scalar(1.0 - self.beta1);
            let v = self.v[i].clone().mul_scalar(self.beta2)
                + g.powi_scalar(2).mul_scalar(1.0 - self.beta2);
            let mhat = m.clone().div_scalar(bc1);
            let vhat = v.clone().div_scalar(bc2);
            self.m[i] = m;
            self.v[i] = v;

            let w = p.inner();
            let decay = w.clone().mul_scalar(self.weight_decay);
            let update = (mhat / (vhat.sqrt() + self.eps) + decay).mul_scalar(self.lr);
            out.push(Tensor::from_inner(w - update).require_grad());
        }
        out
    }
}
