//! Hephaistos — a GPT/Llama implemented from scratch in Rust.
//!
//! Phase 0: project scaffold (`Config`, `matmul`).
//! Phase 1: data & tokenizer (see `data`).
//! Phase 2: GPT-2-style forward pass (see `model`).

mod data;
mod model;

use std::path::Path;

use data::{encode_ids, train_bpe, write_u16_le, DataLoader};
use model::{Config, Gpt};

const CORPUS: &str = "data/input.txt";
const TOK_PATH: &str = "data/tokenizer.json";
const TRAIN_BIN: &str = "data/train.bin";
const VAL_BIN: &str = "data/val.bin";
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
    model.forward(&x);

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
