//! Phase 7 — autoregressive sampling from a trained checkpoint.
//!
//! The forward pass has a fixed `[B, T]` shape, so we generate with a sliding
//! window of exactly `block_size` real tokens: run forward, read the logits at
//! the last position, sample the next token, drop the oldest, repeat. No
//! padding, so attention never sees filler.

use rand::Rng;

use crate::model::Gpt;

/// Generate `n_new` tokens continuing `seed` (which must be `block_size` tokens).
/// The model must have `batch_size == 1`.
pub fn generate<R: Rng>(
    model: &mut Gpt,
    seed: &[u16],
    n_new: usize,
    temperature: f32,
    top_k: Option<usize>,
    rng: &mut R,
) -> Vec<u16> {
    let t = model.cfg.block_size;
    let v = model.cfg.vocab_size;
    assert_eq!(model.cfg.batch_size, 1, "sampling model must have batch_size 1");
    assert_eq!(seed.len(), t, "seed must be block_size tokens");

    let mut window = seed.to_vec();
    let mut out = Vec::with_capacity(n_new);
    for _ in 0..n_new {
        model.forward(&window, None);
        let logits = &model.logits()[(t - 1) * v..t * v];
        let next = sample_logits(logits, temperature, top_k, rng);
        out.push(next);
        window.remove(0);
        window.push(next);
    }
    out
}

/// Sample one token id from raw `logits` with temperature and optional top-k.
fn sample_logits<R: Rng>(logits: &[f32], temperature: f32, top_k: Option<usize>, rng: &mut R) -> u16 {
    let v = logits.len();
    let mut s: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();

    // top-k: keep only the k largest logits
    if let Some(k) = top_k {
        if k < v {
            let mut idx: Vec<usize> = (0..v).collect();
            idx.sort_unstable_by(|&a, &b| s[b].total_cmp(&s[a]));
            for &i in &idx[k..] {
                s[i] = f32::NEG_INFINITY;
            }
        }
    }

    // softmax
    let maxv = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in s.iter_mut() {
        *x = (*x - maxv).exp();
        sum += *x;
    }

    // sample from the cumulative distribution
    let r = rng.r#gen::<f32>() * sum;
    let mut acc = 0.0f32;
    for (i, &p) in s.iter().enumerate() {
        acc += p;
        if acc >= r {
            return i as u16;
        }
    }
    (v - 1) as u16
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Config;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn generate_produces_valid_tokens() {
        let cfg = Config {
            n_layer: 2,
            n_head: 2,
            n_embd: 16,
            block_size: 8,
            vocab_size: 32,
            batch_size: 1,
        };
        let mut rng = StdRng::seed_from_u64(0);
        let mut model = Gpt::new(cfg, &mut rng);
        let seed: Vec<u16> = (0..cfg.block_size as u16).collect();
        let out = generate(&mut model, &seed, 20, 0.8, Some(10), &mut rng);
        assert_eq!(out.len(), 20);
        assert!(out.iter().all(|&id| (id as usize) < cfg.vocab_size));
    }
}
