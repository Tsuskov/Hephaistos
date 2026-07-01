//! Phase 1 — data & tokenizer.
//!
//! Train a fresh byte-level BPE on the corpus (via the from-scratch `cadmus`
//! crate), dump token ids as flat `u16` binaries, and hand out random `[B, T]`
//! batches.

use std::fs::{self, File};
use std::io::{BufWriter, Read, Write};

use cadmus::BpeModel;
use rand::Rng;

/// Train a byte-level BPE on `corpus_path`, save it as an HF `tokenizer.json`
/// (the shape `gguf.rs` reads) to `out_path`, and return the model ready for
/// encode/decode.
pub fn train_bpe(corpus_path: &str, out_path: &str, vocab_size: usize) -> std::io::Result<BpeModel> {
    let text = fs::read_to_string(corpus_path)?;
    let model = BpeModel::train(&text, vocab_size, 2);
    fs::write(out_path, model.to_hf_json())?;
    Ok(model)
}

/// Encode `text` to a flat `u16` id stream (vocab fits in u16 for our configs).
pub fn encode_ids(tok: &BpeModel, text: &str) -> Vec<u16> {
    tok.encode(text).iter().map(|&id| id as u16).collect()
}

/// Write ids as little-endian `u16` bytes.
pub fn write_u16_le(path: &str, ids: &[u16]) -> std::io::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    for &id in ids {
        w.write_all(&id.to_le_bytes())?;
    }
    w.flush()
}

/// Holds a token stream in memory and serves random `[B, T]` batches.
pub struct DataLoader {
    data: Vec<u16>,
    pub batch_size: usize,
    pub block_size: usize,
}

impl DataLoader {
    pub fn from_bin(path: &str, batch_size: usize, block_size: usize) -> std::io::Result<Self> {
        let mut bytes = Vec::new();
        File::open(path)?.read_to_end(&mut bytes)?;
        assert!(bytes.len() % 2 == 0, "bin file must be u16-aligned");
        let data: Vec<u16> = bytes
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        assert!(
            data.len() > block_size + 1,
            "corpus has {} tokens, need > {}",
            data.len(),
            block_size + 1
        );
        Ok(Self { data, batch_size, block_size })
    }

    /// Returns `(inputs, targets)`, each a flat `[batch_size * block_size]`
    /// row-major buffer. `targets` is `inputs` shifted one token to the left.
    pub fn next_batch<R: Rng>(&self, rng: &mut R) -> (Vec<u16>, Vec<u16>) {
        let (b, t) = (self.batch_size, self.block_size);
        let mut x = Vec::with_capacity(b * t);
        let mut y = Vec::with_capacity(b * t);
        let max_start = self.data.len() - t - 1;
        for _ in 0..b {
            let i = rng.gen_range(0..=max_start);
            x.extend_from_slice(&self.data[i..i + t]);
            y.extend_from_slice(&self.data[i + 1..i + 1 + t]);
        }
        (x, y)
    }

    pub fn num_tokens(&self) -> usize {
        self.data.len()
    }
}
