//! Phase 4 — the gradient-check harness ("truth machine").
//!
//! Built *before* the backward pass: "loss goes down" does NOT prove gradients
//! are correct. Every analytic op in Phase 5 gets checked against a central-
//! difference numerical gradient here.
//!
//! We validate the harness itself against the one loss gradient we can write
//! analytically before any backward exists: `d loss / d logits = (probs - 1{j=t}) / N`.

use crate::model::Gpt;

/// Relative error `|a-b| / max(|a|,|b|, tiny)`. Below ~1e-4 means "matches".
pub fn rel_error(a: f64, b: f64) -> f64 {
    let denom = a.abs().max(b.abs()).max(1e-12);
    (a - b).abs() / denom
}

/// Central-difference gradient of the loss w.r.t. flat parameter `i`, evaluated
/// with the f64 forward so round-off doesn't floor the accuracy at ~1e-3.
/// This is the workhorse Phase 5 uses to check each analytic parameter gradient.
pub fn numerical_grad(model: &mut Gpt, ids: &[u16], targets: &[u16], i: usize, eps: f32) -> f64 {
    let orig = model.param(i);
    // Use the actual f32 perturbation span as the denominator so the parameter's
    // f32 granularity doesn't leak into the estimate.
    let plus = orig + eps;
    let minus = orig - eps;
    model.set_param(i, plus);
    let lp = model.loss_f64(ids, targets);
    model.set_param(i, minus);
    let lm = model.loss_f64(ids, targets);
    model.set_param(i, orig);
    (lp - lm) / (plus as f64 - minus as f64)
}

/// Mean cross-entropy computed directly from a `logits [n, v]` buffer, in f64.
/// The only loss path we can differentiate analytically pre-backward, so it is
/// our yardstick for the harness.
fn loss_from_logits(logits: &[f32], targets: &[u16], n: usize, v: usize) -> f64 {
    let mut total = 0.0f64;
    for r in 0..n {
        let row = &logits[r * v..(r + 1) * v];
        let maxv = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        let mut sum = 0.0f64;
        for &x in row {
            sum += ((x as f64) - maxv).exp();
        }
        let pt = ((row[targets[r] as usize] as f64) - maxv).exp() / sum;
        total += -pt.ln();
    }
    total / n as f64
}

/// Central-difference gradient of `loss_from_logits` w.r.t. `logits[idx]` (f64).
fn numerical_logit_grad(
    logits: &mut [f32],
    targets: &[u16],
    n: usize,
    v: usize,
    idx: usize,
    eps: f32,
) -> f64 {
    let orig = logits[idx];
    logits[idx] = orig + eps;
    let lp = loss_from_logits(logits, targets, n, v);
    logits[idx] = orig - eps;
    let lm = loss_from_logits(logits, targets, n, v);
    logits[idx] = orig;
    (lp - lm) / (2.0 * eps as f64)
}

/// Validate the harness end-to-end: compare the analytic softmax+CE logit
/// gradient `(probs - onehot)/N` against the central-difference estimate over
/// every logit. Returns the maximum relative error (should be ~1e-6).
pub fn validate_softmax_ce(model: &mut Gpt, ids: &[u16], targets: &[u16], eps: f32) -> f64 {
    let cfg = model.cfg;
    let n = cfg.batch_size * cfg.block_size;
    let v = cfg.vocab_size;

    model.forward(ids, Some(targets));
    let mut logits = model.logits().to_vec();

    // Analytic gradients from an f64 softmax of the same logits (precomputed so
    // the buffer isn't borrowed while we perturb it numerically).
    let mut analytic = vec![0.0f64; n * v];
    for r in 0..n {
        let row = &logits[r * v..(r + 1) * v];
        let maxv = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        let sum: f64 = row.iter().map(|&x| ((x as f64) - maxv).exp()).sum();
        for j in 0..v {
            let p = ((row[j] as f64) - maxv).exp() / sum;
            let onehot = if j == targets[r] as usize { 1.0 } else { 0.0 };
            analytic[r * v + j] = (p - onehot) / n as f64;
        }
    }

    let mut max_rel = 0.0f64;
    for idx in 0..n * v {
        let numeric = numerical_logit_grad(&mut logits, targets, n, v, idx, eps);
        max_rel = max_rel.max(rel_error(analytic[idx], numeric));
    }
    max_rel
}

/// Run forward+backward, then for each parameter tensor sample up to
/// `per_tensor` entries and compare the analytic gradient against the f64
/// central-difference estimate. Returns `(name, max_rel_error, n_checked)` per
/// tensor. Entries with negligible gradient (both < 1e-8) are skipped.
pub fn check_gradients(
    model: &mut Gpt,
    ids: &[u16],
    targets: &[u16],
    eps: f32,
    per_tensor: usize,
) -> Vec<(&'static str, f64, usize)> {
    // f64 analytic gradients (formula correctness, isolated from f32 round-off).
    let analytic = model.grads_f64(ids, targets);

    let spans = model.param_spans();
    let mut out = Vec::with_capacity(spans.len());
    for (name, off, len) in spans {
        let step = (len / per_tensor).max(1);
        let mut max_rel = 0.0f64;
        let mut checked = 0usize;
        let mut i = 0;
        while i < len {
            let idx = off + i;
            let a = analytic[idx];
            let numeric = numerical_grad(model, ids, targets, idx, eps);
            if a.abs().max(numeric.abs()) > 1e-8 {
                max_rel = max_rel.max(rel_error(a, numeric));
                checked += 1;
            }
            i += step;
        }
        out.push((name, max_rel, checked));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Config;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn mini() -> Config {
        Config {
            n_layer: 2,
            n_head: 2,
            n_embd: 16,
            block_size: 8,
            vocab_size: 32,
            batch_size: 2,
        }
    }

    #[test]
    fn harness_matches_softmax_ce_analytic_grad() {
        let cfg = mini();
        let mut rng = StdRng::seed_from_u64(7);
        let mut model = Gpt::new(cfg, &mut rng);
        let n = cfg.batch_size * cfg.block_size;
        let ids: Vec<u16> = (0..n).map(|i| (i % cfg.vocab_size) as u16).collect();
        let targets: Vec<u16> = (0..n).map(|i| ((i * 5 + 1) % cfg.vocab_size) as u16).collect();

        let max_rel = validate_softmax_ce(&mut model, &ids, &targets, 1e-3);
        assert!(max_rel < 1e-4, "harness max rel error {max_rel} too high");
    }

    #[test]
    fn all_param_grads_match_numerical() {
        let cfg = mini();
        let mut rng = StdRng::seed_from_u64(123);
        let mut model = Gpt::new(cfg, &mut rng);
        let n = cfg.batch_size * cfg.block_size;
        let ids: Vec<u16> = (0..n).map(|i| ((i * 13 + 2) % cfg.vocab_size) as u16).collect();
        let targets: Vec<u16> = (0..n).map(|i| ((i * 7 + 1) % cfg.vocab_size) as u16).collect();

        let results = check_gradients(&mut model, &ids, &targets, 1e-5, 8);
        for (name, max_rel, checked) in results {
            assert!(checked > 0, "{name}: nothing checked");
            assert!(max_rel < 1e-4, "{name}: max rel error {max_rel} >= 1e-4");
        }
    }

    #[test]
    fn numerical_param_grad_is_finite() {
        let cfg = mini();
        let mut rng = StdRng::seed_from_u64(1);
        let mut model = Gpt::new(cfg, &mut rng);
        let n = cfg.batch_size * cfg.block_size;
        let ids: Vec<u16> = (0..n).map(|i| (i % cfg.vocab_size) as u16).collect();
        let targets: Vec<u16> = (0..n).map(|i| ((i * 3) % cfg.vocab_size) as u16).collect();

        let g = numerical_grad(&mut model, &ids, &targets, 0, 1e-3);
        assert!(g.is_finite());
    }
}
