//! Hephaistos — a GPT/Llama implemented from scratch in Rust.
//!
//! Phase 0: project scaffold (`Config`, `matmul`).
//! Phase 1: data & tokenizer (see `data`).
//! Phase 2: GPT-2-style forward pass (see `model`).
//! Phase 4: gradient-check harness (see `gradcheck`).

mod data;
mod gradcheck;
mod model;
mod sample;
mod train;

use std::path::Path;

use data::{encode_ids, train_bpe, write_u16_le, DataLoader};
use model::{Config, Gpt};

const CORPUS: &str = "data/input.txt";
const TOK_PATH: &str = "data/tokenizer.json";
const TRAIN_BIN: &str = "data/train.bin";
const VAL_BIN: &str = "data/val.bin";
const CKPT: &str = "data/ckpt.bin";
const VOCAB_SIZE: usize = 1024;
const BATCH_SIZE: usize = 4;
const BLOCK_SIZE: usize = 64;

// Phase-2 demo model size (real Config arrives in Phase 8).
const N_LAYER: usize = 4;
const N_HEAD: usize = 4;
const N_EMBD: usize = 128;

/// Row-major matrix multiply: `out[m x n] = a[m x k] * b[k x n]`.
///
/// All matrices are flat, row-major slices. `out` must have length `m * n`.
/// This is the deliberately-naive triple loop; Phase 9 replaces it with a fast
/// path, but the reference semantics live here.
pub fn matmul(out: &mut [f32], a: &[f32], b: &[f32], m: usize, k: usize, n: usize) {
    assert_eq!(a.len(), m * k, "a must be m*k");
    assert_eq!(b.len(), k * n, "b must be k*n");
    assert_eq!(out.len(), m * n, "out must be m*n");

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            out[i * n + j] = sum;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Phase 8: real Llama run on the Greek corpus vs the MLX reference curve.
    // Gated behind an env var so the default run stays the phase-0..7 demo.
    if std::env::var("PHASE8").is_ok() {
        return phase8();
    }

    // Phase 0 sanity.
    let mut out = [0.0f32; 4];
    matmul(&mut out, &[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0], 2, 2, 2);
    println!("matmul sanity [[1,2],[3,4]]x[[5,6],[7,8]] = {out:?}");

    // Phase 1: tokenizer (train once, then reuse the saved one).
    let tok = if Path::new(TOK_PATH).exists() {
        println!("loading tokenizer from {TOK_PATH}");
        tokenizers::Tokenizer::from_file(TOK_PATH)?
    } else {
        println!("training byte-level BPE (vocab={VOCAB_SIZE}) on {CORPUS} ...");
        train_bpe(CORPUS, TOK_PATH, VOCAB_SIZE)?
    };
    println!("vocab size = {}", tok.get_vocab_size(true));

    // Encode corpus -> train/val bins (once).
    if !(Path::new(TRAIN_BIN).exists() && Path::new(VAL_BIN).exists()) {
        let text = std::fs::read_to_string(CORPUS)?;
        let split = text.len() * 9 / 10; // tinyshakespeare is ASCII -> byte split is safe
        let (train_text, val_text) = text.split_at(split);
        let train_ids = encode_ids(&tok, train_text)?;
        let val_ids = encode_ids(&tok, val_text)?;
        write_u16_le(TRAIN_BIN, &train_ids)?;
        write_u16_le(VAL_BIN, &val_ids)?;
        println!("wrote {} train + {} val tokens", train_ids.len(), val_ids.len());
    }

    // Fertig-wenn: pull a batch, decode it, see readable text.
    let loader = DataLoader::from_bin(TRAIN_BIN, BATCH_SIZE, BLOCK_SIZE)?;
    println!("train.bin = {} tokens", loader.num_tokens());
    let mut rng = rand::thread_rng();
    let (x, y) = loader.next_batch(&mut rng);
    println!("batch x,y each = [{BATCH_SIZE}, {BLOCK_SIZE}] = {} tokens", x.len());

    let row0: Vec<u32> = x[0..BLOCK_SIZE].iter().map(|&t| t as u32).collect();
    let decoded = tok.decode(&row0, false)?;
    println!("\n--- decoded batch row 0 ---\n{decoded}\n---------------------------");

    // Within a row, targets are inputs shifted left by one.
    assert_eq!(&x[1..BLOCK_SIZE], &y[0..BLOCK_SIZE - 1]);
    println!("targets == inputs shifted by 1 ✓");

    // Phase 2: build a GPT-2-style model and run one forward pass.
    let cfg = Config {
        n_layer: N_LAYER,
        n_head: N_HEAD,
        n_embd: N_EMBD,
        block_size: BLOCK_SIZE,
        vocab_size: VOCAB_SIZE,
        batch_size: BATCH_SIZE,
    };
    let mut model = Gpt::new(cfg, &mut rng);
    println!("\nmodel: {} params", model.num_params());
    let loss = model.forward(&x, Some(&y)).unwrap();

    let logits = model.logits();
    assert_eq!(logits.len(), BATCH_SIZE * BLOCK_SIZE * VOCAB_SIZE);
    let (mut mn, mut mx, mut sum) = (f32::INFINITY, f32::NEG_INFINITY, 0.0f64);
    for &v in logits {
        assert!(v.is_finite(), "logit not finite");
        mn = mn.min(v);
        mx = mx.max(v);
        sum += v as f64;
    }
    println!(
        "logits shape = [{BATCH_SIZE}, {BLOCK_SIZE}, {VOCAB_SIZE}] = {} values",
        logits.len()
    );
    println!(
        "logits: min {mn:.4}, max {mx:.4}, mean {:.4} (all finite ✓)",
        sum / logits.len() as f64
    );

    // argmax over the last position of batch row 0 -> a (random, untrained) prediction
    let last = &logits[(BLOCK_SIZE - 1) * VOCAB_SIZE..BLOCK_SIZE * VOCAB_SIZE];
    let argmax = last
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .unwrap();
    let pred = tok.decode(&[argmax], false)?;
    println!("untrained next-token prediction (row 0): id {argmax} = {pred:?}");

    // Phase 3: untrained loss should sit near ln(vocab_size) (uniform-guess).
    let expected = (VOCAB_SIZE as f32).ln();
    println!("loss = {loss:.4}  (expected ~ln({VOCAB_SIZE}) = {expected:.4})");

    // Phase 4: gradient-check harness on a mini-config, validated against the
    // analytic softmax+CE logit gradient (the only one we can write pre-backward).
    let mini = Config {
        n_layer: 2,
        n_head: 2,
        n_embd: 16,
        block_size: 8,
        vocab_size: 32,
        batch_size: 2,
    };
    let mut mini_model = Gpt::new(mini, &mut rng);
    let mn = mini.batch_size * mini.block_size;
    let ids2: Vec<u16> = (0..mn).map(|i| (i % mini.vocab_size) as u16).collect();
    let tgt2: Vec<u16> = (0..mn).map(|i| ((i * 5 + 1) % mini.vocab_size) as u16).collect();

    let max_rel = gradcheck::validate_softmax_ce(&mut mini_model, &ids2, &tgt2, 1e-3);
    println!("\nPhase 4 harness: softmax+CE analytic vs numerical, max rel error = {max_rel:.2e}");

    // Phase 5: full backward, every parameter tensor gradient-checked
    // (f64 analytic vs f64 numerical). Also exercise the f32 training backward.
    mini_model.forward(&ids2, Some(&tgt2));
    mini_model.backward(&ids2, &tgt2);
    let _ = mini_model.grad(0); // f32 param grads populated (used by Phase 6)
    println!("Phase 5 gradient check (max rel error vs f64 numerical, < 1e-4 = correct):");
    let results = gradcheck::check_gradients(&mut mini_model, &ids2, &tgt2, 1e-5, 8);
    let mut worst = 0.0f64;
    for (name, rel, checked) in &results {
        println!("  {name:9} {rel:.2e}  ({checked} checked)");
        worst = worst.max(*rel);
    }
    println!("worst tensor: {worst:.2e}");

    // Phase 6: train the demo model on tinyshakespeare and watch the loss drop.
    let val_loader = DataLoader::from_bin(VAL_BIN, BATCH_SIZE, BLOCK_SIZE)?;
    let mut train_model = Gpt::new(cfg, &mut rng);
    let tcfg = train::TrainConfig {
        steps: 300,
        lr: 1e-3,
        weight_decay: 0.1,
        eval_every: 25,
        eval_batches: 5,
        patience: 100,
        ckpt_path: CKPT.to_string(),
    };
    println!("\nPhase 6 training (start loss ~ln(1024) = {:.2}):", (VOCAB_SIZE as f32).ln());
    let t0 = std::time::Instant::now();
    let best = train::train(&mut train_model, &loader, &val_loader, &mut rng, &tcfg)?;
    let dt = t0.elapsed().as_secs_f64();
    println!("trained {} steps in {dt:.2}s ({:.1} steps/s)", tcfg.steps, tcfg.steps as f64 / dt);
    train_model.load_params(CKPT)?; // restore best-val weights
    println!("best val loss = {best:.4}  (restored from {CKPT})");

    // Phase 7: sample from the trained checkpoint. Params are independent of
    // batch_size, so build a batch_size=1 twin and load the same weights.
    let sample_cfg = Config { batch_size: 1, ..cfg };
    let mut sample_model = Gpt::new(sample_cfg, &mut rng);
    sample_model.load_params(CKPT)?;

    let (seed_batch, _) = loader.next_batch(&mut rng);
    let seed = &seed_batch[0..BLOCK_SIZE]; // a real contiguous snippet
    let generated = sample::generate(&mut sample_model, seed, 200, 0.8, Some(40), &mut rng);

    let seed_txt = tok.decode(&seed.iter().map(|&t| t as u32).collect::<Vec<_>>(), false)?;
    let gen_txt = tok.decode(&generated.iter().map(|&t| t as u32).collect::<Vec<_>>(), false)?;
    println!("\nPhase 7 sample (temperature 0.8, top-k 40):");
    println!("--- seed ---\n{seed_txt}");
    println!("--- generated ---\n{gen_txt}");

    Ok(())
}

/// Phase 8 — train the from-scratch Llama on the Greek corpus with the exact
/// Posaidon hyperparameters and compare the loss trajectory to the MLX reference
/// (`Posaidon/out_greek/train.log`). Reuses Posaidon's BPE tokenizer (vocab 2048)
/// so the token stream matches, and splits tokens 90/10 like the reference.
fn phase8() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use rand::SeedableRng;

    const GREEK_TOK: &str = "data/greek_tokenizer.json";
    const GREEK_CORPUS: &str = "data/greek_input.txt";
    const GREEK_TRAIN: &str = "data/greek_train.bin";
    const GREEK_VAL: &str = "data/greek_val.bin";
    const GREEK_CKPT: &str = "data/greek_ckpt.bin";

    let tok = tokenizers::Tokenizer::from_file(GREEK_TOK)?;
    println!("Phase 8: Greek run | tokenizer vocab = {}", tok.get_vocab_size(true));

    // Encode the whole corpus, then split tokens 90/10 (matches the reference).
    if !(Path::new(GREEK_TRAIN).exists() && Path::new(GREEK_VAL).exists()) {
        let text = std::fs::read_to_string(GREEK_CORPUS)?;
        let ids = encode_ids(&tok, &text)?;
        let split = ids.len() * 9 / 10;
        write_u16_le(GREEK_TRAIN, &ids[..split])?;
        write_u16_le(GREEK_VAL, &ids[split..])?;
        println!("encoded {} tokens -> train {} / val {}", ids.len(), split, ids.len() - split);
    }

    let cfg = Config {
        n_layer: 8,
        n_head: 8,
        n_embd: 384,
        block_size: 256,
        vocab_size: 2048,
        batch_size: 32,
    };
    let mut rng = rand::rngs::StdRng::seed_from_u64(1337);
    let mut model = Gpt::new(cfg, &mut rng);
    model.set_dropout(0.1);
    println!("model: {} params (reference: 15.74M)", model.num_params());

    let loader = DataLoader::from_bin(GREEK_TRAIN, cfg.batch_size, cfg.block_size)?;
    let val_loader = DataLoader::from_bin(GREEK_VAL, cfg.batch_size, cfg.block_size)?;

    // Short validation run by default; override step count with PHASE8_STEPS.
    let steps: usize = std::env::var("PHASE8_STEPS").ok().and_then(|s| s.parse().ok()).unwrap_or(250);
    let tcfg = train::TrainConfig {
        steps,
        lr: 3e-4,
        weight_decay: 0.1,
        eval_every: 50,
        eval_batches: 5,
        patience: 100000, // disable early stop for the validation run
        ckpt_path: GREEK_CKPT.to_string(),
    };
    println!("training {steps} steps (lr 3e-4, wd 0.1, dropout 0.1)");
    println!("MLX reference: iter250 t4.63/v5.04, iter500 t3.88/v4.46, val-min 3.97 @ 1500");
    let t0 = std::time::Instant::now();
    let best = train::train(&mut model, &loader, &val_loader, &mut rng, &tcfg)?;
    let dt = t0.elapsed().as_secs_f64();
    println!("done: {steps} steps in {dt:.1}s ({:.2} steps/s), best val {best:.4}", steps as f64 / dt);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_hand_checked_2x2() {
        // [[1,2],[3,4]] * [[5,6],[7,8]]
        //  = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
        //  = [[19, 22], [43, 50]]
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let mut out = [0.0f32; 4];
        matmul(&mut out, &a, &b, 2, 2, 2);
        assert_eq!(out, [19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn matmul_non_square_2x3_times_3x2() {
        // A (2x3) = [[1,2,3],[4,5,6]]
        // B (3x2) = [[7,8],[9,10],[11,12]]
        // AB = [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
        //    = [[58, 64], [139, 154]]
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut out = [0.0f32; 4];
        matmul(&mut out, &a, &b, 2, 3, 2);
        assert_eq!(out, [58.0, 64.0, 139.0, 154.0]);
    }
}
