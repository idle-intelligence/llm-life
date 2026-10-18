//! The three models against the classical rule on the CPU and, where an
//! adapter exists, on WebGPU. Weights are the published checkpoints in the
//! life-models layout (tests/fixtures).

use life::{Grid, Rule};
use life_small::{Backend, Small};

const MODELS: [(&str, &[u8], &str); 3] = [
    ("bert", include_bytes!("fixtures/bert/model.safetensors"), include_str!("fixtures/bert/config.json")),
    ("mlp2", include_bytes!("fixtures/mlp2/model.safetensors"), include_str!("fixtures/mlp2/config.json")),
    ("stencil", include_bytes!("fixtures/stencil/model.safetensors"), include_str!("fixtures/stencil/config.json")),
];

fn board(w: usize, h: usize, seed: u64) -> Vec<u8> {
    let mut s = seed | 1;
    (0..w * h)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) % 100 < 30) as u8
        })
        .collect()
}

fn truth(cells: &[u8], w: usize, h: usize) -> Vec<u8> {
    let g = Grid::from_cells(w, h, cells.to_vec());
    g.step(&Rule::life()).cells().to_vec()
}

fn alive(p: &[f32]) -> Vec<u8> {
    p.iter().map(|&p| (p >= 0.5) as u8).collect()
}

fn has_adapter() -> bool {
    let inst = wgpu::Instance::new(&Default::default());
    pollster::block_on(inst.request_adapter(&Default::default())).is_ok()
}

fn sizes() -> [(usize, usize); 4] {
    [(16, 16), (32, 32), (24, 40), (64, 64)]
}

#[test]
fn cpu_boards_match_the_rule() {
    for (name, st, cfg) in MODELS {
        let m = pollster::block_on(Small::new(st, cfg, false)).unwrap();
        for (w, h) in sizes() {
            if name != "stencil" && (w, h) != (16, 16) && (w, h) != (32, 32) {
                continue;
            }
            let b = board(w, h, 7 + w as u64);
            let p = pollster::block_on(m.p_alive(&b, w, h)).unwrap();
            assert_eq!(alive(&p), truth(&b, w, h), "{name} {w}x{h}");
        }
    }
}

#[test]
fn per_cell_equals_whole_board() {
    for (name, st, cfg) in MODELS {
        let m = pollster::block_on(Small::new(st, cfg, false)).unwrap();
        let (w, h) = (16, 12);
        let b = board(w, h, 3);
        let all = pollster::block_on(m.p_alive(&b, w, h)).unwrap();
        for (i, expect) in all.iter().enumerate() {
            let one = pollster::block_on(m.p_alive_cell(&b, w, h, i)).unwrap();
            assert!((one - expect).abs() < 1e-5, "{name} cell {i}: {one} vs {expect}");
        }
    }
}

#[test]
fn gpu_matches_cpu() {
    if !has_adapter() {
        eprintln!("no WebGPU adapter: skipped");
        return;
    }
    for (name, st, cfg) in MODELS {
        let cpu = pollster::block_on(Small::new(st, cfg, false)).unwrap();
        let gpu = pollster::block_on(Small::new(st, cfg, true)).unwrap();
        assert!(matches!(gpu.backend(), Backend::Gpu(_)), "{name}");
        for (w, h) in [(16, 16), (32, 32), (64, 64)] {
            let b = board(w, h, 11);
            let a = pollster::block_on(cpu.logits(&b, w, h)).unwrap();
            let g = pollster::block_on(gpu.logits(&b, w, h)).unwrap();
            let d = a.iter().zip(&g).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max);
            assert!(d < 1e-4, "{name} {w}x{h}: max logit diff {d}");
            let p = pollster::block_on(gpu.p_alive(&b, w, h)).unwrap();
            assert_eq!(alive(&p), truth(&b, w, h), "{name} gpu {w}x{h}");
        }
    }
}
