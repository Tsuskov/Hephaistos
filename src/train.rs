//! Phase 6 — training loop (AdamW) with validation, best-checkpointing and
//! early stopping.

use rand::Rng;

use crate::data::DataLoader;
use crate::model::Gpt;

pub struct TrainConfig {
    pub steps: usize,
    pub lr: f32,
    pub weight_decay: f32,
    pub eval_every: usize,
    pub eval_batches: usize,
    pub patience: usize, // stop after this many evals without val improvement
    pub ckpt_path: String,
}

/// Mean forward loss over `batches` validation batches (no backward).
pub fn evaluate<R: Rng>(model: &mut Gpt, val: &DataLoader, rng: &mut R, batches: usize) -> f32 {
    let mut total = 0.0f32;
    for _ in 0..batches {
        let (x, y) = val.next_batch(rng);
        total += model.forward(&x, Some(&y)).unwrap();
    }
    total / batches as f32
}

/// Train `model` in place. Returns the best validation loss seen (its weights
/// are saved to `cfg.ckpt_path`).
pub fn train<R: Rng>(
    model: &mut Gpt,
    train: &DataLoader,
    val: &DataLoader,
    rng: &mut R,
    cfg: &TrainConfig,
) -> std::io::Result<f32> {
    let (beta1, beta2, eps) = (0.9f32, 0.999f32, 1e-8f32);
    let mut best = f32::INFINITY;
    let mut since_improved = 0usize;

    for step in 1..=cfg.steps {
        let (x, y) = train.next_batch(rng);
        let loss = model.forward(&x, Some(&y)).unwrap();
        model.backward(&x, &y);
        model.adamw_step(cfg.lr, beta1, beta2, eps, cfg.weight_decay);

        if step == 1 || step % cfg.eval_every == 0 {
            let vl = evaluate(model, val, rng, cfg.eval_batches);
            println!("step {step:4}: train {loss:.4}  val {vl:.4}");
            if vl < best {
                best = vl;
                since_improved = 0;
                model.save(&cfg.ckpt_path)?;
            } else {
                since_improved += 1;
                if since_improved >= cfg.patience {
                    println!("early stop: no val improvement for {} evals", cfg.patience);
                    break;
                }
            }
        }
    }
    Ok(best)
}
