//! The three small Game of Life models, each run over a board on the best
//! available backend: WebGPU when an adapter is granted, otherwise one CPU
//! thread. The choice is made once, when the model is created, from whether a
//! device could be had; it is never measured. Both backends read the same
//! weights and give the same logits to within float rounding.

use life_models::{gpu::GpuModel, Kind, Model};

pub use life_models::Error;

#[cfg(feature = "web")]
pub mod web;

pub enum Backend {
    Gpu(Box<GpuModel>),
    Cpu,
}

pub struct Small {
    model: Model,
    backend: Backend,
}

impl Small {
    /// `safetensors` and `config.json` as in the life-models checkpoint
    /// layout. With `use_gpu` the WebGPU backend is tried first and the CPU
    /// is taken when there is no adapter or the device is refused.
    pub async fn new(safetensors: &[u8], config_json: &str, use_gpu: bool) -> Result<Small, Error> {
        let model = Model::from_bytes(safetensors, config_json)?;
        let backend = if use_gpu {
            match model.to_gpu().await {
                Ok(g) => Backend::Gpu(Box::new(g)),
                Err(_) => Backend::Cpu,
            }
        } else {
            Backend::Cpu
        };
        Ok(Small { model, backend })
    }

    pub fn kind(&self) -> Kind {
        self.model.kind()
    }

    pub fn backend(&self) -> &Backend {
        &self.backend
    }

    /// "WebGPU" with the adapter, or "CPU, 1 thread".
    pub fn label(&self) -> String {
        match &self.backend {
            Backend::Gpu(g) => format!("WebGPU, {}", g.adapter()),
            Backend::Cpu => "CPU, 1 thread".to_string(),
        }
    }

    /// Logits for every cell of a toroidal `w * h` board.
    pub async fn logits(&self, cells: &[u8], w: usize, h: usize) -> Result<Vec<f32>, Error> {
        match &self.backend {
            Backend::Gpu(g) => g.logits(cells, w, h).await,
            Backend::Cpu => self.model.logits(cells, w, h),
        }
    }

    /// p(alive) for every cell: softmax over `[dead, alive]` for the two
    /// token-free classifiers, sigmoid of the single logit for the stencil.
    pub async fn p_alive(&self, cells: &[u8], w: usize, h: usize) -> Result<Vec<f32>, Error> {
        let lg = self.logits(cells, w, h).await?;
        Ok(p_from_logits(self.kind(), &lg))
    }

    /// p(alive) of one cell. The cell's 3x3 neighbourhood is cut out of the
    /// torus and run as a 3x3 torus of its own, whose centre sees exactly
    /// the same eight neighbours.
    pub async fn p_alive_cell(&self, cells: &[u8], w: usize, h: usize, index: usize) -> Result<f32, Error> {
        if w == 0 || h == 0 || cells.len() != w * h || index >= cells.len() {
            return Err(Error::Board(format!("cell {index} of a {w}x{h} board with {} cells", cells.len())));
        }
        let (x, y) = ((index % w) as isize, (index / w) as isize);
        let at = |dx: isize, dy: isize| {
            let xx = (x + dx).rem_euclid(w as isize) as usize;
            let yy = (y + dy).rem_euclid(h as isize) as usize;
            cells[yy * w + xx]
        };
        let mut patch = [0u8; 9];
        for dy in -1..=1isize {
            for dx in -1..=1isize {
                patch[((dy + 1) * 3 + dx + 1) as usize] = at(dx, dy);
            }
        }
        let lg = self.logits(&patch, 3, 3).await?;
        let k = self.kind().outputs();
        Ok(p_from_logits(self.kind(), &lg[4 * k..5 * k])[0])
    }
}

pub fn p_from_logits(kind: Kind, lg: &[f32]) -> Vec<f32> {
    match kind {
        Kind::Bert | Kind::Mlp2 => lg.chunks_exact(2).map(|l| sigmoid(l[1] - l[0])).collect(),
        Kind::Stencil => lg.iter().map(|&l| sigmoid(l)).collect(),
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
