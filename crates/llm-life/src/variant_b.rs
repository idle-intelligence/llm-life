//! Variant B: one language model whose attention is restricted to a 2D
//! stencil (CONCEPT.md §2).
//!
//! The grid *is* the context. One token per cell, serialized row-major after
//! a shared rules prefix; token `i` attends to the prefix, to itself and to
//! its 8 grid neighbors, and to nothing else. One forward pass is one
//! generation for every cell at once, and the logits at cell `i` are the
//! model's prediction of cell `i`'s next state.
//!
//! Positions: **bag mode** by default (CONCEPT.md §4). Life is
//! outer-totalistic — only "my state" and "how many neighbors are alive"
//! matter, never which neighbor — so every grid token gets the same position
//! id. RoPE is relative, so all pairwise rotations between grid tokens
//! vanish and the neighborhood becomes an unordered bag. Self-vs-neighbor is
//! carried by the mask and by the query token itself, not by position.

use life::{Grid, Rule};

/// The shared rules prefix. Deliberately plain text with the rulestring
/// quoted in it: CONCEPT.md §5's held-out-rules experiment needs the rule to
/// live in the prompt, not in the weights.
pub fn rules_prefix(rule: &Rule) -> String {
    format!(
        "Cellular automaton, rule {}. Each cell is 0 (dead) or 1 (alive).\n\
         A live cell with 2 or 3 live neighbors stays 1, otherwise it becomes 0.\n\
         A dead cell with exactly 3 live neighbors becomes 1, otherwise it stays 0.\n\
         Grid:\n",
        rule.to_rulestring()
    )
}

/// The rules prefix with variant A's six worked examples inserted just before
/// `Grid:`.
///
/// The examples are variant A's, **verbatim** (`Neighbors: … / Self: … /
/// Next: N`). They cannot be written in the form variant B's cells take: a
/// cell here is a single bare digit token whose "self" and "neighbors" roles
/// are carried by the stencil mask and by nothing else, so there is no text
/// that is one cell. This is the closest form, and it is the same text that
/// moved variant A off answering `0` everywhere.
pub fn fewshot_rules_prefix(rule: &Rule) -> String {
    let p = rules_prefix(rule);
    let head = p.strip_suffix("Grid:\n").unwrap();
    format!("{head}{}Grid:\n", crate::variant_a::fewshot_examples())
}

/// One packed forward pass: `prefix ++ one token per cell`.
pub struct Packed {
    /// prefix tokens followed by `width * height` cell tokens.
    pub tokens: Vec<u32>,
    /// One RoPE position per token.
    pub positions: Vec<u32>,
    /// Row-major `[T, T]`, `true` where query `i` may attend to key `j`.
    pub allowed: Vec<bool>,
    /// Index of the first cell token in `tokens`.
    pub grid_start: usize,
}

impl Packed {
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

/// Build the packed sequence, positions and stencil mask for one generation.
///
/// `prefix_tokens` is the tokenized [`rules_prefix`]; `dead`/`alive` are the
/// single-token ids for `0` and `1`.
pub fn pack(grid: &Grid, prefix_tokens: &[u32], dead: u32, alive: u32) -> Packed {
    let p = prefix_tokens.len();
    let n = grid.width() * grid.height();
    let t = p + n;

    let mut tokens = Vec::with_capacity(t);
    tokens.extend_from_slice(prefix_tokens);
    tokens.extend(grid.cells().iter().map(|&c| if c != 0 { alive } else { dead }));

    // Prefix keeps real positions (it is ordinary text and must read as
    // such); every cell token shares position `p` — the bag.
    let mut positions: Vec<u32> = (0..p as u32).collect();
    positions.extend(std::iter::repeat_n(p as u32, n));

    // The mask is the whole point, so it is built explicitly rather than
    // derived: prefix rows are causal among themselves, cell rows see the
    // prefix plus their own 9-cell neighborhood and nothing else. In
    // particular a cell row does NOT see other cells' rows even though they
    // are "earlier" in the sequence.
    let mut allowed = vec![false; t * t];
    for i in 0..p {
        for j in 0..=i {
            allowed[i * t + j] = true;
        }
    }
    for c in 0..n {
        let row = (p + c) * t;
        for j in 0..p {
            allowed[row + j] = true;
        }
        allowed[row + p + c] = true;
        for j in grid.neighbor_indices(c) {
            allowed[row + p + j] = true;
        }
    }

    Packed {
        tokens,
        positions,
        allowed,
        grid_start: p,
    }
}

/// p(alive) per cell from the sliced head's `[T, 2]` logits (column 0 =
/// `dead`, column 1 = `alive`), read at the cell rows only.
///
/// Softmax over just the two answer tokens: the question asked is "given
/// that the answer is 0 or 1, which is it", so renormalizing over the
/// answer vocabulary is the right reading, and it is also what makes the
/// grayscale picture the model's *confidence* rather than its absolute
/// probability of emitting a digit at all.
pub fn p_alive(logits: &[f32], grid_start: usize, n_cells: usize) -> Vec<f32> {
    (0..n_cells)
        .map(|c| {
            let d = logits[(grid_start + c) * 2];
            let a = logits[(grid_start + c) * 2 + 1];
            let m = d.max(a);
            let (ed, ea) = ((d - m).exp(), (a - m).exp());
            ea / (ed + ea)
        })
        .collect()
}

/// Threshold p(alive) at 0.5 — the argmax grid.
pub fn argmax_grid(p: &[f32], width: usize, height: usize) -> Grid {
    Grid::from_cells(width, height, p.iter().map(|&v| (v >= 0.5) as u8).collect())
}

/// The same packed sequence as [`pack`], with the stencil expressed as
/// llm-web's `SparseMask` (a contiguous key prefix plus an explicit key
/// list per query) instead of a dense `[T, T]` bool mask.
///
/// This is the form the mask has always had — a cell reads the rules prefix
/// and its 9-cell neighbourhood — and the dense form was only ever a
/// transcription of it. At 64x64 that transcription is a 17M-entry
/// `Vec<bool>`; at 128x128 it is 271 MB on the CPU and the same again on
/// the GPU, before a single attention score is computed. Here it is
/// `T * (2 + 9)` u32 either way.
///
/// [`pack`] stays: it is what `tests/variant_b.rs` checks the topology
/// against and what llm-web's `tests/stencil.rs` uses as the numerical
/// oracle for the kernel.
pub struct PackedSparse {
    pub tokens: Vec<u32>,
    pub positions: Vec<u32>,
    /// Per query, the length of the attended contiguous key range `[0, .)`:
    /// `i + 1` for a prefix row (causal), the whole prefix for a cell row.
    pub prefix_len: Vec<u32>,
    /// Per query, how many of its `keys` row is valid.
    pub n_keys: Vec<u32>,
    /// Row-major `[T, MAX_STENCIL_KEYS]`.
    pub keys: Vec<u32>,
    pub grid_start: usize,
}

/// Self plus 8 neighbours — the widest explicit key list a cell row has.
pub const MAX_STENCIL_KEYS: usize = 9;

pub fn pack_sparse(grid: &Grid, prefix_tokens: &[u32], dead: u32, alive: u32) -> PackedSparse {
    let p = prefix_tokens.len();
    let n = grid.width() * grid.height();
    let t = p + n;

    let mut tokens = Vec::with_capacity(t);
    tokens.extend_from_slice(prefix_tokens);
    tokens.extend(grid.cells().iter().map(|&c| if c != 0 { alive } else { dead }));

    let mut positions: Vec<u32> = (0..p as u32).collect();
    positions.extend(std::iter::repeat_n(p as u32, n));

    let mut prefix_len: Vec<u32> = (1..=p as u32).collect();
    prefix_len.extend(std::iter::repeat_n(p as u32, n));
    let mut n_keys = vec![0u32; t];
    let mut keys = vec![0u32; t * MAX_STENCIL_KEYS];
    for c in 0..n {
        let row = p + c;
        let base = row * MAX_STENCIL_KEYS;
        keys[base] = (p + c) as u32;
        let mut k = 1;
        for j in grid.neighbor_indices(c) {
            keys[base + k] = (p + j) as u32;
            k += 1;
        }
        n_keys[row] = k as u32;
    }

    PackedSparse {
        tokens,
        positions,
        prefix_len,
        n_keys,
        keys,
        grid_start: p,
    }
}
