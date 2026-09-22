//! Minimal binary PGM (P5) writer — the pictures in `docs/pictures/`.
//!
//! PGM because it is a dozen lines of code with no dependency and every
//! viewer and every Python script reads it; the demo page paints from the
//! same float grids directly.

use anyhow::{Context, Result};
use std::io::Write;
use std::path::Path;

/// Write `values` (0.0..=1.0, row-major) as an 8-bit grayscale PGM, scaled up
/// by `zoom` so a 64x64 grid is legible without a viewer that can zoom.
/// 0.0 is white and 1.0 is black, matching the demo page.
pub fn write_pgm(path: &Path, values: &[f32], width: usize, height: usize, zoom: usize) -> Result<()> {
    anyhow::ensure!(values.len() == width * height, "write_pgm: values.len() does not match width * height");
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

/// Read back a PGM written by [`write_pgm`]/[`write_binary_pgm`] at a known
/// `zoom` (every `zoom x zoom` block is one cell — read its top-left pixel)
/// and undo the `0.0 = white, 1.0 = black` convention, returning one float
/// per cell at the original `width x height`.
pub fn read_pgm(path: &Path, zoom: usize) -> Result<(Vec<f32>, usize, usize)> {
    let bytes = std::fs::read(path)?;
    anyhow::ensure!(bytes.starts_with(b"P5\n"), "{}: not a binary PGM", path.display());
    let mut fields = bytes[3..].splitn(2, |&b| b == b'\n');
    let dims = fields.next().context("pgm: missing dimensions")?;
    let rest = fields.next().context("pgm: missing maxval")?;
    let dims = std::str::from_utf8(dims)?;
    let mut dims = dims.split_whitespace();
    let zoomed_w: usize = dims.next().context("pgm: missing width")?.parse()?;
    let zoomed_h: usize = dims.next().context("pgm: missing height")?.parse()?;

    let mut lines = rest.splitn(2, |&b| b == b'\n');
    let _maxval = lines.next().context("pgm: missing maxval line")?;
    let data = lines.next().context("pgm: missing pixel data")?;
    anyhow::ensure!(data.len() == zoomed_w * zoomed_h, "{}: truncated pixel data", path.display());

    let (width, height) = (zoomed_w / zoom, zoomed_h / zoom);
    let mut values = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            let g = data[(y * zoom) * zoomed_w + x * zoom];
            values.push(1.0 - g as f32 / 255.0);
        }
    }
    Ok((values, width, height))
}
