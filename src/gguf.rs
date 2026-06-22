//! Phase 11 — minimal GGUF v3 writer.
//!
//! Just enough of the format to emit a llama-architecture model that
//! llama.cpp / Ollama can load: a header, typed metadata key/values, tensor
//! infos, and 32-byte-aligned F32 tensor data. The model→GGUF tensor mapping and
//! the RoPE permute live in `model::Gpt::export_gguf`; this file only knows the
//! byte layout.

use std::collections::HashSet;
use std::fs::File;
use std::io::Write;

const ALIGNMENT: usize = 32;

// GGUF metadata value type tags.
const T_UINT32: u32 = 4;
const T_INT32: u32 = 5;
const T_FLOAT32: u32 = 6;
const T_BOOL: u32 = 7;
const T_STRING: u32 = 8;
const T_ARRAY: u32 = 9;

const GGML_TYPE_F32: u32 = 0;

enum Kv {
    U32(u32),
    F32(f32),
    Bool(bool),
    Str(String),
    ArrStr(Vec<String>),
    ArrI32(Vec<i32>),
}

struct Tensor {
    name: String,
    dims: Vec<u64>, // GGUF ne order (innermost/contiguous first)
    data: Vec<u8>,  // little-endian f32 bytes
}

/// Accumulates metadata and tensors, then writes one GGUF v3 file.
pub struct GgufWriter {
    kvs: Vec<(String, Kv)>,
    tensors: Vec<Tensor>,
}

impl GgufWriter {
    pub fn new() -> Self {
        Self { kvs: Vec::new(), tensors: Vec::new() }
    }

    pub fn kv_u32(&mut self, k: &str, v: u32) {
        self.kvs.push((k.to_string(), Kv::U32(v)));
    }
    pub fn kv_f32(&mut self, k: &str, v: f32) {
        self.kvs.push((k.to_string(), Kv::F32(v)));
    }
    pub fn kv_bool(&mut self, k: &str, v: bool) {
        self.kvs.push((k.to_string(), Kv::Bool(v)));
    }
    pub fn kv_str(&mut self, k: &str, v: &str) {
        self.kvs.push((k.to_string(), Kv::Str(v.to_string())));
    }
    pub fn kv_arr_str(&mut self, k: &str, v: Vec<String>) {
        self.kvs.push((k.to_string(), Kv::ArrStr(v)));
    }
    pub fn kv_arr_i32(&mut self, k: &str, v: Vec<i32>) {
        self.kvs.push((k.to_string(), Kv::ArrI32(v)));
    }

    /// Add an F32 tensor. `dims` is in GGUF ne order: for a row-major
    /// `[rows, cols]` weight, pass `[cols, rows]` and the raw row-major data.
    pub fn add_tensor(&mut self, name: &str, dims: Vec<u64>, data: &[f32]) {
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for &x in data {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        self.tensors.push(Tensor { name: name.to_string(), dims, data: bytes });
    }

    pub fn write(&self, path: &str) -> std::io::Result<()> {
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        buf.extend_from_slice(&(self.kvs.len() as u64).to_le_bytes());

        for (k, v) in &self.kvs {
            write_string(&mut buf, k);
            write_value(&mut buf, v);
        }

        // Tensor data offsets are relative to the (aligned) data section start,
        // each tensor padded to ALIGNMENT.
        let mut offsets = Vec::with_capacity(self.tensors.len());
        let mut off = 0usize;
        for t in &self.tensors {
            offsets.push(off as u64);
            off = align_up(off + t.data.len(), ALIGNMENT);
        }

        for (t, &offset) in self.tensors.iter().zip(&offsets) {
            write_string(&mut buf, &t.name);
            buf.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
            for &d in &t.dims {
                buf.extend_from_slice(&d.to_le_bytes());
            }
            buf.extend_from_slice(&GGML_TYPE_F32.to_le_bytes());
            buf.extend_from_slice(&offset.to_le_bytes());
        }

        // Pad to the data section, then write each tensor at its offset.
        while buf.len() % ALIGNMENT != 0 {
            buf.push(0);
        }
        let data_start = buf.len();
        for (t, &offset) in self.tensors.iter().zip(&offsets) {
            let target = data_start + offset as usize;
            while buf.len() < target {
                buf.push(0);
            }
            buf.extend_from_slice(&t.data);
        }

        File::create(path)?.write_all(&buf)
    }
}

fn write_string(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

fn write_value(buf: &mut Vec<u8>, v: &Kv) {
    match v {
        Kv::U32(x) => {
            buf.extend_from_slice(&T_UINT32.to_le_bytes());
            buf.extend_from_slice(&x.to_le_bytes());
        }
        Kv::F32(x) => {
            buf.extend_from_slice(&T_FLOAT32.to_le_bytes());
            buf.extend_from_slice(&x.to_le_bytes());
        }
        Kv::Bool(b) => {
            buf.extend_from_slice(&T_BOOL.to_le_bytes());
            buf.push(*b as u8);
        }
        Kv::Str(s) => {
            buf.extend_from_slice(&T_STRING.to_le_bytes());
            write_string(buf, s);
        }
        Kv::ArrStr(items) => {
            buf.extend_from_slice(&T_ARRAY.to_le_bytes());
            buf.extend_from_slice(&T_STRING.to_le_bytes());
            buf.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for s in items {
                write_string(buf, s);
            }
        }
        Kv::ArrI32(items) => {
            buf.extend_from_slice(&T_ARRAY.to_le_bytes());
            buf.extend_from_slice(&T_INT32.to_le_bytes());
            buf.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for x in items {
                buf.extend_from_slice(&x.to_le_bytes());
            }
        }
    }
}

fn align_up(x: usize, a: usize) -> usize {
    x.div_ceil(a) * a
}

/// Parse an HF `tokenizer.json` (byte-level BPE) and add the GGUF `gpt2`
/// tokenizer metadata: token list, token types, merges, and the special-token
/// ids. Mirrors what llama.cpp's HF converter writes for this tokenizer kind.
pub fn add_byte_bpe_tokenizer(w: &mut GgufWriter, tokenizer_path: &str) -> std::io::Result<()> {
    let json: serde_json::Value = serde_json::from_reader(File::open(tokenizer_path)?)?;
    let model = &json["model"];

    let vocab = model["vocab"].as_object().expect("tokenizer.model.vocab");
    let mut tokens = vec![String::new(); vocab.len()];
    for (tok, id) in vocab {
        tokens[id.as_u64().unwrap() as usize] = tok.clone();
    }

    let mut special: HashSet<String> = HashSet::new();
    if let Some(added) = json["added_tokens"].as_array() {
        for a in added {
            if a["special"].as_bool() == Some(true) {
                if let Some(c) = a["content"].as_str() {
                    special.insert(c.to_string());
                }
            }
        }
    }

    // GGUF TokenType: NORMAL=1, UNKNOWN=2, CONTROL=3.
    let types: Vec<i32> = tokens
        .iter()
        .map(|t| if t == "<unk>" { 2 } else if special.contains(t) { 3 } else { 1 })
        .collect();

    // merges may be ["a b", ...] or [["a","b"], ...] depending on tokenizer version.
    let merges: Vec<String> = model["merges"]
        .as_array()
        .expect("tokenizer.model.merges")
        .iter()
        .map(|m| match m.as_array() {
            Some(pair) => format!("{} {}", pair[0].as_str().unwrap(), pair[1].as_str().unwrap()),
            None => m.as_str().unwrap().to_string(),
        })
        .collect();

    let unk = vocab["<unk>"].as_u64().expect("tokenizer needs <unk>") as u32;

    w.kv_str("tokenizer.ggml.model", "gpt2");
    w.kv_str("tokenizer.ggml.pre", "default");
    w.kv_arr_str("tokenizer.ggml.tokens", tokens);
    w.kv_arr_i32("tokenizer.ggml.token_type", types);
    w.kv_arr_str("tokenizer.ggml.merges", merges);
    w.kv_u32("tokenizer.ggml.unknown_token_id", unk);
    w.kv_u32("tokenizer.ggml.bos_token_id", unk);
    w.kv_u32("tokenizer.ggml.eos_token_id", unk);
    w.kv_bool("tokenizer.ggml.add_bos_token", false);
    Ok(())
}
