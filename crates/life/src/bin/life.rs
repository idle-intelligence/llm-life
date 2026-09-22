//! Command-line classical Game of Life: prints the grid after each
//! generation as `#`/`.`, no external dependencies, no LLM anywhere.

use life::{Grid, Rule};

fn print_grid(grid: &Grid, generation: usize) {
    println!("gen {} alive {}", generation, grid.live_count());
    for y in 0..grid.height() {
        let mut row = String::with_capacity(grid.width());
        for x in 0..grid.width() {
            row.push(if grid.get(x, y) != 0 { '#' } else { '.' });
        }
        println!("{row}");
    }
}

fn main() {
    let mut size = 32usize;
    let mut rulestring = "B3/S23".to_string();
    let mut seed: Option<u64> = None;
    let mut glider = false;
    let mut generations = 10usize;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--size" => size = args.next().expect("--size needs a value").parse().expect("--size must be an integer"),
            "--rule" => rulestring = args.next().expect("--rule needs a value"),
            "--seed" => seed = Some(args.next().expect("--seed needs a value").parse().expect("--seed must be an integer")),
            "--glider" => glider = true,
            "--generations" => {
                generations = args.next().expect("--generations needs a value").parse().expect("--generations must be an integer")
            }
            other => panic!("unknown argument: {other}"),
        }
    }

    let rule = Rule::parse(&rulestring).unwrap_or_else(|| panic!("bad rulestring: {rulestring}"));

    let mut grid = if glider {
        let mut g = Grid::new(size, size);
        g.place_glider(0, 0);
        g
    } else {
        Grid::random(size, size, seed.unwrap_or(0), 0.3)
    };

    print_grid(&grid, 0);
    for gen in 1..=generations {
        grid = grid.step(&rule);
        print_grid(&grid, gen);
    }
}
