//! B/S rulestrings: the rule as data, not as code.
//!
//! `CONCEPT.md` §5 wants a *family* of outer-totalistic rules (the rule lives
//! in the prompt prefix, not in the weights), so the classical engine parses
//! the same rulestring the LLM prefix will quote.

/// An outer-totalistic 2-state rule, e.g. `B3/S23` (Conway's Life).
///
/// `birth[n]` / `survive[n]` are indexed by live-neighbor count 0..=8.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rule {
    pub birth: [bool; 9],
    pub survive: [bool; 9],
}

impl Rule {
    /// Conway's Life.
    pub fn life() -> Self {
        Rule::parse("B3/S23").unwrap()
    }

    /// Parse `B<digits>/S<digits>` (case-insensitive). Returns `None` on any
    /// malformed input; digits outside 0..=8 are malformed.
    pub fn parse(s: &str) -> Option<Self> {
        let (b, sv) = s.split_once('/')?;
        let b = b.strip_prefix(['B', 'b'])?;
        let sv = sv.strip_prefix(['S', 's'])?;
        let mut rule = Rule {
            birth: [false; 9],
            survive: [false; 9],
        };
        for c in b.chars() {
            rule.birth[c.to_digit(9)? as usize] = true;
        }
        for c in sv.chars() {
            rule.survive[c.to_digit(9)? as usize] = true;
        }
        Some(rule)
    }

    /// Canonical rulestring, e.g. `B3/S23`.
    pub fn to_rulestring(&self) -> String {
        let mut s = String::from("B");
        for (n, &on) in self.birth.iter().enumerate() {
            if on {
                s.push(char::from_digit(n as u32, 10).unwrap());
            }
        }
        s.push_str("/S");
        for (n, &on) in self.survive.iter().enumerate() {
            if on {
                s.push(char::from_digit(n as u32, 10).unwrap());
            }
        }
        s
    }

    /// Next state of one cell from its own state and its live-neighbor count.
    pub fn next(&self, alive: bool, neighbors: usize) -> bool {
        if alive {
            self.survive[neighbors]
        } else {
            self.birth[neighbors]
        }
    }
}
