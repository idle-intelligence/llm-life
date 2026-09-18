//! Variant A: 4096 language models called in parallel (CONCEPT.md §2 A).
//!
//! Each cell gets its own little prompt — `Neighbors: … / Self: … / Next:` —
//! and we read the answer logits at the final `:` of that prompt. The 4096
//! prompts are not batched: they are *packed* into one long sequence with a
//! block-diagonal attention mask, so a cell's tokens see the shared rules
//! prefix and their own block and nothing else. There is no batch axis
//! anywhere, which is what lets this run on an engine that asserts batch 1.
//!
//! Positions: the prefix keeps `0..P`, and every cell block restarts at `P`.
//! RoPE is relative, so each cell sees exactly the same geometry it would see
//! if it were the only prompt in the sequence — the packing is numerically
//! invisible to the model, which is the whole point of doing it this way
//! rather than concatenating prompts and hoping.
//!
//! The prefix is prefilled once per generation and left resident in the KV
//! cache; chunks are forwarded against it and the cache is rewound to the
//! prefix length afterwards (`KvCache::snapshot`/`restore`).

use life::{Grid, Rule};

/// The shared rules prefix. Unlike variant B's, it states the answer format,
/// because variant A's per-cell prompt is an explicit question.
pub fn rules_prefix(rule: &Rule) -> String {
    format!(
        "Cellular automaton, rule {}. Each cell is 0 (dead) or 1 (alive).\n\
         A live cell with 2 or 3 live neighbors stays 1, otherwise it becomes 0.\n\
         A dead cell with exactly 3 live neighbors becomes 1, otherwise it stays 0.\n\
         For each cell, answer with one digit: its next state.\n",
        rule.to_rulestring()
    )
}

/// Six worked examples covering birth, survival and both deaths, appended to
/// the prefix by `--fewshot`. The instruct model answers `0` almost
/// everywhere from the rules alone (docs/OVERNIGHT-REPORT.md); this is the
/// cheapest thing that could make `1` reachable.
pub fn fewshot_examples() -> String {
    let cases: [([u8; 8], u8, u8); 6] = [
        ([1, 1, 1, 0, 0, 0, 0, 0], 0, 1), // birth: dead, exactly 3
        ([1, 1, 0, 1, 0, 0, 0, 0], 1, 1), // survival: alive, 3
        ([1, 1, 0, 0, 0, 0, 0, 0], 1, 1), // survival: alive, 2
        ([1, 1, 1, 1, 0, 0, 0, 0], 1, 0), // overcrowding: alive, 4
        ([1, 0, 0, 0, 0, 0, 0, 0], 1, 0), // loneliness: alive, 1
        ([1, 1, 0, 0, 0, 0, 0, 0], 0, 0), // dead, 2: stays dead
    ];
    let mut s = String::from("Examples:\n");
    for (nb, me, next) in cases {
        s.push_str(&cell_prompt(&nb, me));
        s.push_str(&format!("{next}\n"));
    }
    s
}

/// One cell's prompt. It ends with a **trailing space**, deliberately: Qwen's
/// pre-tokenizer splits digits off and leaves the leading space as its own
/// token, so `"Next:"` would be followed by `" "` and only then by the digit.
/// Ending the prompt on the space puts the bare `0`/`1` token at the very
/// next position, which is the position whose logits we read.
pub fn cell_prompt(neighbors: &[u8], self_state: u8) -> String {
    let mut s = String::from("Neighbors:");
    for &n in neighbors {
        s.push(' ');
        s.push(if n != 0 { '1' } else { '0' });
    }
    s.push_str(" / Self: ");
    s.push(if self_state != 0 { '1' } else { '0' });
    s.push_str(" / Next: ");
    s
}

/// Where one cell's tokens live inside a chunk.
#[derive(Debug, Clone, Copy)]
pub struct Block {
    /// Index of the cell in the grid.
    pub cell: usize,
    /// Offset of the block's first token within the chunk.
    pub start: usize,
    pub len: usize,
}

/// One chunk of cells, packed against a resident prefix of `prefix_len`.
pub struct Chunk {
    pub tokens: Vec<u32>,
    pub positions: Vec<u32>,
    /// Row-major `[T, prefix_len + T]`, `true` where query `i` may attend to
    /// key `j`. Keys `0..prefix_len` are the resident prefix.
    pub allowed: Vec<bool>,
    pub blocks: Vec<Block>,
    pub prefix_len: usize,
}

impl Chunk {
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Row index (within the chunk) whose logits answer each cell — the last
    /// token of the block, i.e. the final `:`.
    pub fn answer_rows(&self) -> Vec<usize> {
        self.blocks.iter().map(|b| b.start + b.len - 1).collect()
    }
}

/// Pack `cells` (grid indices) into one chunk. `suffix` maps a cell index to
/// its already-tokenized prompt.
pub fn pack_chunk(cells: &[usize], suffix: impl Fn(usize) -> Vec<u32>, prefix_len: usize) -> Chunk {
    let mut tokens = Vec::new();
    let mut positions = Vec::new();
    let mut blocks = Vec::with_capacity(cells.len());

    for &cell in cells {
        let ids = suffix(cell);
        let start = tokens.len();
        positions.extend((0..ids.len()).map(|s| (prefix_len + s) as u32));
        tokens.extend_from_slice(&ids);
        blocks.push(Block {
            cell,
            start,
            len: ids.len(),
        });
    }

    let t = tokens.len();
    let kv = prefix_len + t;
    let mut allowed = vec![false; t * kv];
    for b in &blocks {
        for i in 0..b.len {
            let row = (b.start + i) * kv;
            // The resident prefix, in full.
            allowed[row..row + prefix_len].fill(true);
            // Causal *within the block only*: this is the block-diagonal part.
            let base = row + prefix_len + b.start;
            allowed[base..base + i + 1].fill(true);
        }
    }

    Chunk {
        tokens,
        positions,
        allowed,
        blocks,
        prefix_len,
    }
}

/// Every cell of the grid, as the per-cell prompt text.
pub fn cell_prompts(grid: &Grid) -> Vec<String> {
    (0..grid.cells().len())
        .map(|c| {
            let nb: Vec<u8> = grid
                .neighbor_indices(c)
                .iter()
                .map(|&j| grid.cells()[j])
                .collect();
            cell_prompt(&nb, grid.cells()[c])
        })
        .collect()
}

/// p(alive) for the cells of one chunk, from that chunk's `[T, 2]` sliced
/// logits (column 0 = `dead`, column 1 = `alive`).
pub fn p_alive_chunk(logits: &[f32], chunk: &Chunk) -> Vec<(usize, f32)> {
    chunk
        .blocks
        .iter()
        .map(|b| {
            let r = b.start + b.len - 1;
            let (d, a) = (logits[r * 2], logits[r * 2 + 1]);
            let m = d.max(a);
            let (ed, ea) = ((d - m).exp(), (a - m).exp());
            (b.cell, ea / (ed + ea))
        })
        .collect()
}
