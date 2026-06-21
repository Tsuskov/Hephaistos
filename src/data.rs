//! Phase 1 — data & tokenizer.
//!
//! Train a fresh byte-level BPE on the corpus, dump token ids as flat `u16`
//! binaries, and hand out random `[B, T]` batches.

use std::fs::File;
use std::io::{BufWriter, Read, Write};

use rand::Rng;
use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{NormalizerWrapper, Tokenizer, TokenizerBuilder, TokenizerImpl};

/// Train a byte-level BPE on `corpus_path`, save it to `out_path`, and return
/// it as a type-erased `Tokenizer` ready for encode/decode.
pub fn train_bpe(
    corpus_path: &str,
    out_path: &str,
    vocab_size: usize,
) -> tokenizers::Result<Tokenizer> {
    let mut trainer = BpeTrainerBuilder::new()
        .show_progress(false)
        .vocab_size(vocab_size)
        .min_frequency(2)
        // Seed all 256 byte-level chars so nothing is ever <unk>.
        .initial_alphabet(ByteLevel::alphabet().into_iter().collect())
        .build();

    let mut tokenizer: TokenizerImpl<BPE, NormalizerWrapper, ByteLevel, ByteLevel, ByteLevel> =
        TokenizerBuilder::new()
            .with_model(BPE::default())
            .with_normalizer(None)
            .with_pre_tokenizer(Some(ByteLevel::default()))
            .with_post_processor(Some(ByteLevel::default()))
            .with_decoder(Some(ByteLevel::default()))
            .build()?;

    tokenizer.train_from_files(&mut trainer, vec![corpus_path.to_string()])?;
    tokenizer.save(out_path, true)?;

    Tokenizer::from_file(out_path)
}

/// Encode `text` to a flat `u16` id stream (vocab fits in u16 for our configs).
pub fn encode_ids(tok: &Tokenizer, text: &str) -> tokenizers::Result<Vec<u16>> {
    let enc = tok.encode(text, false)?;
    Ok(enc.get_ids().iter().map(|&id| id as u16).collect())
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
