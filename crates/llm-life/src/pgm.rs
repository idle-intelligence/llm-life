//! Minimal binary PGM (P5) writer — the pictures in `docs/pictures/`.
//!
//! PGM because it is a dozen lines of code with no dependency and every
//! viewer and every Python script reads it; the demo page paints from the
//! same float grids directly.

use anyhow::Result;
use std::io::Write;
use std::path::Path;

/// Write `values` (0.0..=1.0, row-major) as an 8-bit grayscale PGM, scaled up
/// by `zoom` so a 64x64 grid is legible without a viewer that can zoom.
/// 0.0 is white and 1.0 is black, matching the demo page.
pub fn write_pgm(path: &Path, values: &[f32], width: usize, height: usize, zoom: usize) -> Result<()> {
    assert_eq!(values.len(), width * height);
    let mut out = Vec::with_capacity(width * height * zoom * zoom + 32);
    write!(out, "P5\n{} {}\n255\n", width * zoom, height * zoom)?;
    for y in 0..height {
        let mut row = Vec::with_capacity(width * zoom);
        for x in 0..width {
            let v = values[y * width + x].clamp(0.0, 1.0);
            let g = (255.0 * (1.0 - v)).round() as u8;
            row.extend(std::iter::repeat_n(g, zoom));
        }
        for _ in 0..zoom {
            out.extend_from_slice(&row);
        }
    }
    std::fs::write(path, out)?;
    Ok(())
}

pub fn write_binary_pgm(path: &Path, cells: &[u8], width: usize, height: usize, zoom: usize) -> Result<()> {
    let values: Vec<f32> = cells.iter().map(|&c| c as f32).collect();
    write_pgm(path, &values, width, height, zoom)
}
