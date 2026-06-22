//! Phase 2 — GPT-2-style forward pass.
//!
//! Layout follows Karpathy's `llm.c` so results can be cross-checked: one flat
//! parameter buffer plus named offsets, and an activation "arena" that stores
//! every intermediate the Phase-5 backward will reuse (softmax `att`, layernorm
//! `mean`/`rstd`, pre-GELU `fch`, …).
//!
//! All tensors are flat, row-major `Vec<f32>`. Dimensions:
//! `B` batch, `T` time/block, `C` channels (`n_embd`), `NH` heads,
//! `HS = C/NH` head size, `V` vocab, `L` layers.

use std::fs::File;
use std::io::{Read, Write};

use num_traits::Float;
use rand::Rng;
use rayon::prelude::*;

/// Model hyperparameters. Weights are flat `Vec<f32>` buffers shaped by these.
#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub n_layer: usize,
    pub n_head: usize,
    pub n_embd: usize,
    pub block_size: usize,
    pub vocab_size: usize,
    pub batch_size: usize,
}

/// A `(offset, len)` view into a flat buffer.
#[derive(Clone, Copy)]
struct Off {
    off: usize,
    len: usize,
}

impl Off {
    fn range(&self) -> std::ops::Range<usize> {
        self.off..self.off + self.len
    }
}

/// Offsets of every parameter tensor inside the flat `params` buffer.
#[derive(Clone, Copy)]
struct ParamLayout {
    wte: Off,      // (V, C) token embedding (also tied lm_head weight)
    ln1w: Off,     // (L, C) RMSNorm weight (no bias)
    qkvw: Off,     // (L, 3C, C) no bias (Llama)
    attprojw: Off, // (L, C, C) no bias (Llama)
    ln2w: Off,     // (L, C) RMSNorm weight (no bias)
    w1: Off,       // (L, H, C) SwiGLU gate, no bias
    w3: Off,       // (L, H, C) SwiGLU up, no bias
    w2: Off,       // (L, C, H) SwiGLU down, no bias
    lnfw: Off,     // (C) RMSNorm weight (no bias)
    lm_head: Off,  // (V, C) output projection, untied from wte, no bias
    total: usize,
}

/// SwiGLU hidden width: `int(8/3 · n_embd)`, chosen so the gated FFN has roughly
/// the same parameter count as the 4·n_embd GELU MLP it replaces.
fn swiglu_hidden(ch: usize) -> usize {
    (8.0 / 3.0 * ch as f64) as usize
}

impl ParamLayout {
    fn new(c: &Config) -> Self {
        let (l, v, ch) = (c.n_layer, c.vocab_size, c.n_embd);
        let mut o = 0usize;
        let mut take = |n: usize| {
            let off = o;
            o += n;
            Off { off, len: n }
        };
        let wte = take(v * ch);
        let ln1w = take(l * ch);
        let qkvw = take(l * 3 * ch * ch);
        let attprojw = take(l * ch * ch);
        let ln2w = take(l * ch);
        let h = swiglu_hidden(ch);
        let w1 = take(l * h * ch);
        let w3 = take(l * h * ch);
        let w2 = take(l * ch * h);
        let lnfw = take(ch);
        let lm_head = take(v * ch);
        Self {
            wte, ln1w, qkvw, attprojw, ln2w,
            w1, w3, w2, lnfw, lm_head, total: o,
        }
    }
}

/// Offsets of every activation tensor inside the flat `acts` arena (sized B,T).
#[derive(Clone, Copy)]
struct ActLayout {
    encoded: Off,   // (B, T, C)
    ln1: Off,       // (L, B, T, C)
    ln1_rstd: Off,  // (L, B, T) RMSNorm rstd
    qkv: Off,       // (L, B, T, 3C)
    atty: Off,      // (L, B, T, C)
    preatt: Off,    // (L, B, NH, T, T)
    att: Off,       // (L, B, NH, T, T)
    attproj: Off,   // (L, B, T, C)
    residual2: Off, // (L, B, T, C)
    ln2: Off,       // (L, B, T, C)
    ln2_rstd: Off,  // (L, B, T) RMSNorm rstd
    gate: Off,      // (L, B, T, H) SwiGLU w1·x (pre-silu)
    up: Off,        // (L, B, T, H) SwiGLU w3·x
    glu: Off,       // (L, B, T, H) silu(gate)·up
    fcproj: Off,    // (L, B, T, C)
    residual3: Off, // (L, B, T, C)
    lnf: Off,       // (B, T, C)
    lnf_rstd: Off,  // (B, T) RMSNorm rstd
    logits: Off,    // (B, T, V)
    probs: Off,     // (B, T, V)
    losses: Off,    // (B, T)
    total: usize,
}

impl ActLayout {
    fn new(c: &Config) -> Self {
        let (l, ch, v, nh) = (c.n_layer, c.n_embd, c.vocab_size, c.n_head);
        let (b, t) = (c.batch_size, c.block_size);
        let bt = b * t;
        let mut o = 0usize;
        let mut take = |n: usize| {
            let off = o;
            o += n;
            Off { off, len: n }
        };
        let encoded = take(bt * ch);
        let ln1 = take(l * bt * ch);
        let ln1_rstd = take(l * bt);
        let qkv = take(l * bt * 3 * ch);
        let atty = take(l * bt * ch);
        let preatt = take(l * b * nh * t * t);
        let att = take(l * b * nh * t * t);
        let attproj = take(l * bt * ch);
        let residual2 = take(l * bt * ch);
        let ln2 = take(l * bt * ch);
        let ln2_rstd = take(l * bt);
        let h = swiglu_hidden(ch);
        let gate = take(l * bt * h);
        let up = take(l * bt * h);
        let glu = take(l * bt * h);
        let fcproj = take(l * bt * ch);
        let residual3 = take(l * bt * ch);
        let lnf = take(bt * ch);
        let lnf_rstd = take(bt);
        let logits = take(bt * v);
        let probs = take(bt * v);
        let losses = take(bt);
        Self {
            encoded, ln1, ln1_rstd, qkv, atty, preatt, att, attproj,
            residual2, ln2, ln2_rstd, gate, up, glu, fcproj, residual3,
            lnf, lnf_rstd, logits, probs, losses, total: o,
        }
    }
}

/// Offsets of the dropout masks inside a dedicated `masks` buffer (separate from
/// the activation arena so the read-only masks never alias the activations).
/// Each mask holds the per-element keep factor (`0` or `1/(1-p)`). Dropout sites
/// match Posaidon: token embeddings, attention weights, attn output, MLP output.
#[derive(Clone, Copy)]
struct MaskLayout {
    encoded: Off,  // (B, T, C)
    att: Off,      // (L, B, NH, T, T)
    attproj: Off,  // (L, B, T, C)
    fcproj: Off,   // (L, B, T, C)
    total: usize,
}

impl MaskLayout {
    fn new(c: &Config) -> Self {
        let (l, ch, nh) = (c.n_layer, c.n_embd, c.n_head);
        let (b, t) = (c.batch_size, c.block_size);
        let bt = b * t;
        let mut o = 0usize;
        let mut take = |n: usize| {
            let off = o;
            o += n;
            Off { off, len: n }
        };
        let encoded = take(bt * ch);
        let att = take(l * b * nh * t * t);
        let attproj = take(l * bt * ch);
        let fcproj = take(l * bt * ch);
        Self { encoded, att, attproj, fcproj, total: o }
    }
}

/// A Llama-style transformer: flat params + activation arena, with matching
/// gradient buffers for the backward pass.
pub struct Gpt {
    pub cfg: Config,
    params: Vec<f32>,
    acts: Vec<f32>,
    grads: Vec<f32>,  // param gradients, same layout as `params`
    gacts: Vec<f32>,  // activation gradients, same layout as `acts`
    m: Vec<f32>,      // AdamW first moment, param layout
    v: Vec<f32>,      // AdamW second moment, param layout
    adam_t: u64,      // AdamW timestep
    masks: Vec<f32>,  // dropout keep factors (filled per training step)
    dropout: f32,     // dropout probability (0 disables dropout entirely)
    pl: ParamLayout,
    al: ActLayout,
    ml: MaskLayout,
}

impl Gpt {
    /// Allocate and randomly initialise a model for a fixed `(batch_size, block_size)`.
    pub fn new<R: Rng>(cfg: Config, rng: &mut R) -> Self {
        assert!(cfg.n_embd % cfg.n_head == 0, "n_embd must divide by n_head");
        let pl = ParamLayout::new(&cfg);
        let al = ActLayout::new(&cfg);
        let ml = MaskLayout::new(&cfg);
        let mut params = vec![0.0f32; pl.total];

        // Weights ~ N(0, 0.02); biases 0; LayerNorm weights 1.
        for o in [pl.wte, pl.qkvw, pl.attprojw, pl.w1, pl.w3, pl.w2, pl.lm_head] {
            for x in &mut params[o.range()] {
                *x = randn(rng) * 0.02;
            }
        }
        for o in [pl.ln1w, pl.ln2w, pl.lnfw] {
            for x in &mut params[o.range()] {
                *x = 1.0;
            }
        }

        let acts = vec![0.0f32; al.total];
        let grads = vec![0.0f32; pl.total];
        let gacts = vec![0.0f32; al.total];
        let m = vec![0.0f32; pl.total];
        let v = vec![0.0f32; pl.total];
        Self {
            cfg, params, acts, grads, gacts, m, v, adam_t: 0,
            masks: Vec::new(), dropout: 0.0, pl, al, ml,
        }
    }

    /// Enable dropout with probability `p` (Llama/Posaidon use 0.1). Allocates the
    /// mask buffer. Dropout is applied only by `forward_train`/`backward`; plain
    /// `forward` (eval, sampling, gradient check) stays deterministic.
    pub fn set_dropout(&mut self, p: f32) {
        self.dropout = p;
        self.masks = if p > 0.0 { vec![1.0f32; self.ml.total] } else { Vec::new() };
    }

    pub fn num_params(&self) -> usize {
        self.params.len()
    }

    /// Read a single flat parameter (used by the gradient-check harness).
    pub fn param(&self, i: usize) -> f32 {
        self.params[i]
    }

    /// Overwrite a single flat parameter (used by the gradient-check harness).
    pub fn set_param(&mut self, i: usize, v: f32) {
        self.params[i] = v;
    }

    /// Logits `[B, T, V]` from the last forward pass.
    pub fn logits(&self) -> &[f32] {
        &self.acts[self.al.logits.range()]
    }

    /// Run the forward pass for one batch of token ids (`len == B*T`).
    ///
    /// If `targets` is given, also computes softmax `probs` + per-token
    /// cross-entropy `losses` and returns the mean loss over `B*T`. Activations
    /// are stored in the f32 arena for the Phase-5 backward.
    pub fn forward(&mut self, ids: &[u16], targets: Option<&[u16]>) -> Option<f32> {
        forward_into::<f32>(&self.cfg, &self.params, &mut self.acts, ids, targets, None)
    }

    /// Training forward: samples fresh dropout masks (when dropout is enabled) and
    /// applies them. Must be paired with `backward`, which reuses these masks.
    pub fn forward_train<R: Rng>(
        &mut self,
        ids: &[u16],
        targets: Option<&[u16]>,
        rng: &mut R,
    ) -> Option<f32> {
        if self.dropout > 0.0 {
            self.fill_masks(rng);
            forward_into::<f32>(&self.cfg, &self.params, &mut self.acts, ids, targets, Some(&self.masks))
        } else {
            forward_into::<f32>(&self.cfg, &self.params, &mut self.acts, ids, targets, None)
        }
    }

    /// Sample fresh Bernoulli(keep) dropout masks scaled by `1/keep` (inverted
    /// dropout), so the expected activation is unchanged and eval needs no rescale.
    fn fill_masks<R: Rng>(&mut self, rng: &mut R) {
        let keep = 1.0 - self.dropout;
        let scale = 1.0 / keep;
        for m in &mut self.masks {
            *m = if rng.r#gen::<f32>() < keep { scale } else { 0.0 };
        }
    }

    /// Recompute only the scalar loss in f64 (params cast to f64, fresh f64
    /// activation scratch). The gradient-check harness uses this so numerical
    /// gradients aren't limited by f32 round-off (~1e-3) and can hit < 1e-4.
    pub fn loss_f64(&self, ids: &[u16], targets: &[u16]) -> f64 {
        let params: Vec<f64> = self.params.iter().map(|&x| x as f64).collect();
        let mut acts = vec![0.0f64; self.al.total];
        forward_into::<f64>(&self.cfg, &params, &mut acts, ids, Some(targets), None).unwrap()
    }

    /// Analytic gradient of the loss w.r.t. flat parameter `i` from the last
    /// `backward` call.
    pub fn grad(&self, i: usize) -> f32 {
        self.grads[i]
    }

    /// `(name, offset, len)` of every parameter tensor — used by the gradient
    /// checker to sample per-tensor (so each backward op is exercised).
    pub fn param_spans(&self) -> Vec<(&'static str, usize, usize)> {
        let p = &self.pl;
        vec![
            ("wte", p.wte.off, p.wte.len),
            ("ln1w", p.ln1w.off, p.ln1w.len),
            ("qkvw", p.qkvw.off, p.qkvw.len),
            ("attprojw", p.attprojw.off, p.attprojw.len),
            ("ln2w", p.ln2w.off, p.ln2w.len),
            ("w1", p.w1.off, p.w1.len),
            ("w3", p.w3.off, p.w3.len),
            ("w2", p.w2.off, p.w2.len),
            ("lnfw", p.lnfw.off, p.lnfw.len),
            ("lm_head", p.lm_head.off, p.lm_head.len),
        ]
    }

    /// Backward pass. Requires a prior `forward(ids, Some(targets))`; fills the
    /// f32 `grads` (param) and `gacts` (activation) buffers from zero.
    pub fn backward(&mut self, ids: &[u16], targets: &[u16]) {
        let masks = if self.dropout > 0.0 { Some(self.masks.as_slice()) } else { None };
        backward_into::<f32>(
            &self.cfg, &self.params, &self.acts, &mut self.grads, &mut self.gacts, ids, targets, masks,
        );
    }

    /// Recompute the parameter gradients entirely in f64 (fresh f64 forward +
    /// backward). The gradient checker compares these against f64 numerical
    /// gradients, isolating formula correctness from f32 round-off so the match
    /// lands well under 1e-4.
    pub fn grads_f64(&self, ids: &[u16], targets: &[u16]) -> Vec<f64> {
        let params: Vec<f64> = self.params.iter().map(|&x| x as f64).collect();
        let mut acts = vec![0.0f64; self.al.total];
        forward_into::<f64>(&self.cfg, &params, &mut acts, ids, Some(targets), None);
        let mut grads = vec![0.0f64; self.pl.total];
        let mut gacts = vec![0.0f64; self.al.total];
        backward_into::<f64>(&self.cfg, &params, &acts, &mut grads, &mut gacts, ids, targets, None);
        grads
    }

    /// One AdamW step using the gradients from the last `backward` (decoupled
    /// weight decay). Updates params and the `m`/`v` moment buffers in place.
    pub fn adamw_step(&mut self, lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) {
        self.adam_t += 1;
        let t = self.adam_t as i32;
        let bc1 = 1.0 - beta1.powi(t); // bias corrections
        let bc2 = 1.0 - beta2.powi(t);
        for i in 0..self.params.len() {
            let g = self.grads[i];
            let m = beta1 * self.m[i] + (1.0 - beta1) * g;
            let v = beta2 * self.v[i] + (1.0 - beta2) * g * g;
            self.m[i] = m;
            self.v[i] = v;
            let m_hat = m / bc1;
            let v_hat = v / bc2;
            self.params[i] -= lr * (m_hat / (v_hat.sqrt() + eps) + weight_decay * self.params[i]);
        }
    }

    /// Write parameters to `path` as little-endian f32 (a checkpoint).
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let mut bytes = Vec::with_capacity(self.params.len() * 4);
        for &p in &self.params {
            bytes.extend_from_slice(&p.to_le_bytes());
        }
        File::create(path)?.write_all(&bytes)
    }

    /// Load parameters previously written by `save`.
    pub fn load_params(&mut self, path: &str) -> std::io::Result<()> {
        let mut bytes = Vec::new();
        File::open(path)?.read_to_end(&mut bytes)?;
        assert_eq!(bytes.len(), self.params.len() * 4, "checkpoint size mismatch");
        for (i, c) in bytes.chunks_exact(4).enumerate() {
            self.params[i] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        }
        Ok(())
    }
}

/// The single backward implementation, generic over the float type (mirrors
/// `forward_into`). Zeroes `grads`/`gacts`, then accumulates gradients in
/// reverse order. `acts` must hold the forward activations for these `params`.
fn backward_into<F: Float + Send + Sync>(
    cfg: &Config,
    params: &[F],
    acts: &[F],
    grads: &mut [F],
    gacts: &mut [F],
    ids: &[u16],
    targets: &[u16],
    masks: Option<&[F]>,
) {
    let (b, t, ch, nh, v, nl) = (
        cfg.batch_size, cfg.block_size, cfg.n_embd, cfg.n_head, cfg.vocab_size, cfg.n_layer,
    );
    let bt = b * t;
    let hidden = swiglu_hidden(ch);
    let pl = ParamLayout::new(cfg);
    let al = ActLayout::new(cfg);
    let ml = MaskLayout::new(cfg);

    for x in grads.iter_mut() {
        *x = F::zero();
    }
    for x in gacts.iter_mut() {
        *x = F::zero();
    }

    // loss = mean over B*T tokens -> d loss / d losses[r] = 1/(B*T)
    let dmean = F::one() / cast::<F>(bt as f64);

    crossentropy_softmax_backward(gacts, al.logits.off, acts, al.probs.off, targets, bt, v, dmean);
    // lm_head (logits = lnf @ lm_head^T, untied, no bias)
    linear_backward(
        gacts, al.lnf.off, al.logits.off,
        grads, pl.lm_head.off, None,
        acts, al.lnf.off, params, pl.lm_head.off, bt, ch, v,
    );
    let resf_off = al.residual3.off + (nl - 1) * bt * ch;
    rmsnorm_backward(
        gacts, resf_off, al.lnf.off,
        grads, pl.lnfw.off,
        acts, resf_off, al.lnf_rstd.off,
        params, pl.lnfw.off, bt, ch,
    );

    for l in (0..nl).rev() {
        let res_off = if l == 0 {
            al.encoded.off
        } else {
            al.residual3.off + (l - 1) * bt * ch
        };

        // activation (and activation-grad) offsets
        let ln1_off = al.ln1.off + l * bt * ch;
        let ln1_rstd_off = al.ln1_rstd.off + l * bt;
        let qkv_off = al.qkv.off + l * bt * 3 * ch;
        let atty_off = al.atty.off + l * bt * ch;
        let preatt_off = al.preatt.off + l * b * nh * t * t;
        let att_off = al.att.off + l * b * nh * t * t;
        let attproj_off = al.attproj.off + l * bt * ch;
        let residual2_off = al.residual2.off + l * bt * ch;
        let ln2_off = al.ln2.off + l * bt * ch;
        let ln2_rstd_off = al.ln2_rstd.off + l * bt;
        let gate_off = al.gate.off + l * bt * hidden;
        let up_off = al.up.off + l * bt * hidden;
        let glu_off = al.glu.off + l * bt * hidden;
        let fcproj_off = al.fcproj.off + l * bt * ch;
        let residual3_off = al.residual3.off + l * bt * ch;

        // parameter (and param-grad) offsets
        let ln1w_off = pl.ln1w.off + l * ch;
        let qkvw_off = pl.qkvw.off + l * 3 * ch * ch;
        let attprojw_off = pl.attprojw.off + l * ch * ch;
        let ln2w_off = pl.ln2w.off + l * ch;
        let w1_off = pl.w1.off + l * hidden * ch;
        let w3_off = pl.w3.off + l * hidden * ch;
        let w2_off = pl.w2.off + l * ch * hidden;

        // residual3 = residual2 + fcproj
        residual_backward(gacts, residual2_off, fcproj_off, residual3_off, bt * ch);
        if let Some(m) = masks {
            apply_dropout(gacts, fcproj_off, m, ml.fcproj.off + l * bt * ch, bt * ch);
        }
        // fcproj = glu @ w2^T  (no bias)  -> dglu
        linear_backward(
            gacts, glu_off, fcproj_off,
            grads, w2_off, None,
            acts, glu_off, params, w2_off, bt, hidden, ch,
        );
        // glu = silu(gate) * up  -> dgate, dup
        swiglu_backward(gacts, gate_off, up_off, glu_off, acts, gate_off, up_off, bt * hidden);
        // gate = ln2 @ w1^T  -> accumulates d ln2
        linear_backward(
            gacts, ln2_off, gate_off,
            grads, w1_off, None,
            acts, ln2_off, params, w1_off, bt, ch, hidden,
        );
        // up = ln2 @ w3^T  -> accumulates d ln2 (+=)
        linear_backward(
            gacts, ln2_off, up_off,
            grads, w3_off, None,
            acts, ln2_off, params, w3_off, bt, ch, hidden,
        );
        // ln2 = rmsnorm(residual2)  -> accumulates into d residual2
        rmsnorm_backward(
            gacts, residual2_off, ln2_off,
            grads, ln2w_off,
            acts, residual2_off, ln2_rstd_off,
            params, ln2w_off, bt, ch,
        );
        // residual2 = residual + attproj
        residual_backward(gacts, res_off, attproj_off, residual2_off, bt * ch);
        if let Some(m) = masks {
            apply_dropout(gacts, attproj_off, m, ml.attproj.off + l * bt * ch, bt * ch);
        }
        // attproj = atty @ attprojw^T  (no bias)
        linear_backward(
            gacts, atty_off, attproj_off,
            grads, attprojw_off, None,
            acts, atty_off, params, attprojw_off, bt, ch, ch,
        );
        let att_mask = masks.map(|m| &m[ml.att.off + l * b * nh * t * t..ml.att.off + (l + 1) * b * nh * t * t]);
        attention_backward(
            gacts, qkv_off, preatt_off, att_off, atty_off,
            acts, qkv_off, att_off, b, t, ch, nh, att_mask,
        );
        rope_backward(gacts, qkv_off, b, t, nh, ch / nh);
        // qkv = ln1 @ qkvw^T  (no bias)
        linear_backward(
            gacts, ln1_off, qkv_off,
            grads, qkvw_off, None,
            acts, ln1_off, params, qkvw_off, bt, ch, 3 * ch,
        );
        // ln1 = rmsnorm(residual)  -> accumulates into d residual (layer input)
        rmsnorm_backward(
            gacts, res_off, ln1_off,
            grads, ln1w_off,
            acts, res_off, ln1_rstd_off,
            params, ln1w_off, bt, ch,
        );
    }

    if let Some(m) = masks {
        apply_dropout(gacts, al.encoded.off, m, ml.encoded.off, bt * ch);
    }
    // encoded = wte[ids]  (token-embedding gradient; position is RoPE, no wpe)
    encoder_backward(grads, pl.wte.off, gacts, al.encoded.off, ids, b, t, ch);
}

/// The single forward implementation, generic over the float type so the exact
/// same math runs in f32 (real model, cached for backward) and f64 (the
/// gradient-check reference loss).
fn forward_into<F: Float + Send + Sync>(
    cfg: &Config,
    params: &[F],
    acts: &mut [F],
    ids: &[u16],
    targets: Option<&[u16]>,
    masks: Option<&[F]>,
) -> Option<F> {
    let (b, t, ch, nh, v, nl) = (
        cfg.batch_size, cfg.block_size, cfg.n_embd, cfg.n_head, cfg.vocab_size, cfg.n_layer,
    );
    let bt = b * t;
    let hidden = swiglu_hidden(ch);
    let ml = MaskLayout::new(cfg);
    assert_eq!(ids.len(), bt, "ids must be batch_size * block_size");

    let pl = ParamLayout::new(cfg);
    let al = ActLayout::new(cfg);

    encoder_forward(acts, al.encoded.off, ids, &params[pl.wte.range()], b, t, ch);
    if let Some(m) = masks {
        apply_dropout(acts, al.encoded.off, m, ml.encoded.off, bt * ch);
    }

    for l in 0..nl {
        let res_off = if l == 0 {
            al.encoded.off
        } else {
            al.residual3.off + (l - 1) * bt * ch
        };

        // Per-layer activation offsets.
        let ln1_off = al.ln1.off + l * bt * ch;
        let ln1_rstd_off = al.ln1_rstd.off + l * bt;
        let qkv_off = al.qkv.off + l * bt * 3 * ch;
        let atty_off = al.atty.off + l * bt * ch;
        let preatt_off = al.preatt.off + l * b * nh * t * t;
        let att_off = al.att.off + l * b * nh * t * t;
        let attproj_off = al.attproj.off + l * bt * ch;
        let residual2_off = al.residual2.off + l * bt * ch;
        let ln2_off = al.ln2.off + l * bt * ch;
        let ln2_rstd_off = al.ln2_rstd.off + l * bt;
        let gate_off = al.gate.off + l * bt * hidden;
        let up_off = al.up.off + l * bt * hidden;
        let glu_off = al.glu.off + l * bt * hidden;
        let fcproj_off = al.fcproj.off + l * bt * ch;
        let residual3_off = al.residual3.off + l * bt * ch;

        // Per-layer parameter slices.
        let ln1w = &params[pl.ln1w.off + l * ch..pl.ln1w.off + (l + 1) * ch];
        let qkvw = &params[pl.qkvw.off + l * 3 * ch * ch..pl.qkvw.off + (l + 1) * 3 * ch * ch];
        let attprojw = &params[pl.attprojw.off + l * ch * ch..pl.attprojw.off + (l + 1) * ch * ch];
        let ln2w = &params[pl.ln2w.off + l * ch..pl.ln2w.off + (l + 1) * ch];
        let w1 = &params[pl.w1.off + l * hidden * ch..pl.w1.off + (l + 1) * hidden * ch];
        let w3 = &params[pl.w3.off + l * hidden * ch..pl.w3.off + (l + 1) * hidden * ch];
        let w2 = &params[pl.w2.off + l * ch * hidden..pl.w2.off + (l + 1) * ch * hidden];

        let att_mask = masks.map(|m| &m[ml.att.off + l * b * nh * t * t..ml.att.off + (l + 1) * b * nh * t * t]);

        rmsnorm_forward(acts, ln1_off, ln1_rstd_off, res_off, ln1w, bt, ch);
        linear(acts, qkv_off, ln1_off, qkvw, None, bt, ch, 3 * ch);
        rope_forward(acts, qkv_off, b, t, nh, ch / nh);
        attention_forward(acts, atty_off, preatt_off, att_off, qkv_off, b, t, ch, nh, att_mask);
        linear(acts, attproj_off, atty_off, attprojw, None, bt, ch, ch);
        if let Some(m) = masks {
            apply_dropout(acts, attproj_off, m, ml.attproj.off + l * bt * ch, bt * ch);
        }
        residual_forward(acts, residual2_off, res_off, attproj_off, bt * ch);
        rmsnorm_forward(acts, ln2_off, ln2_rstd_off, residual2_off, ln2w, bt, ch);
        // SwiGLU: glu = silu(ln2 @ w1^T) * (ln2 @ w3^T);  fcproj = glu @ w2^T
        linear(acts, gate_off, ln2_off, w1, None, bt, ch, hidden);
        linear(acts, up_off, ln2_off, w3, None, bt, ch, hidden);
        swiglu_forward(acts, glu_off, gate_off, up_off, bt * hidden);
        linear(acts, fcproj_off, glu_off, w2, None, bt, hidden, ch);
        if let Some(m) = masks {
            apply_dropout(acts, fcproj_off, m, ml.fcproj.off + l * bt * ch, bt * ch);
        }
        residual_forward(acts, residual3_off, residual2_off, fcproj_off, bt * ch);
    }

    let resf_off = al.residual3.off + (nl - 1) * bt * ch;
    rmsnorm_forward(
        acts, al.lnf.off, al.lnf_rstd.off, resf_off,
        &params[pl.lnfw.range()], bt, ch,
    );
    // lm_head: logits = lnf @ lm_head^T, no bias (untied from wte).
    linear(acts, al.logits.off, al.lnf.off, &params[pl.lm_head.range()], None, bt, ch, v);

    targets.map(|targets| {
        assert_eq!(targets.len(), bt, "targets must be batch_size * block_size");
        softmax_forward(acts, al.probs.off, al.logits.off, bt, v);
        crossentropy_forward(acts, al.losses.off, al.probs.off, targets, bt, v);
        let mut sum = F::zero();
        for r in 0..bt {
            sum = sum + acts[al.losses.off + r];
        }
        sum / cast::<F>(bt as f64)
    })
}

/// Standard normal sample via Box–Muller.
fn randn<R: Rng>(rng: &mut R) -> f32 {
    let u1: f32 = rng.r#gen::<f32>().max(1e-7);
    let u2: f32 = rng.r#gen::<f32>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

/// Cast an f64 literal into the generic float type `F`.
#[inline]
fn cast<F: Float>(x: f64) -> F {
    F::from(x).unwrap()
}

fn encoder_forward<F: Float>(
    acts: &mut [F],
    out_off: usize,
    ids: &[u16],
    wte: &[F],
    b: usize,
    t: usize,
    c: usize,
) {
    // Token embedding only; position enters via RoPE inside attention.
    for bi in 0..b {
        for ti in 0..t {
            let ix = ids[bi * t + ti] as usize;
            let o = out_off + (bi * t + ti) * c;
            for ci in 0..c {
                acts[o + ci] = wte[ix * c + ci];
            }
        }
    }
}

/// Rotary position embedding (Llama / MLX non-traditional convention), applied
/// in place to the q and k of each head in the packed `[B, T, 3C]` qkv buffer
/// (q at head offset, k at `C +` head offset; v untouched). For frequency index
/// `j` in `0..hs/2` the pair `(x[j], x[j+hs/2])` is rotated by
/// `angle = pos · 10000^(-2j/hs)`, so position enters multiplicatively instead
/// of via a learned table.
fn rope_forward<F: Float>(acts: &mut [F], qkv_off: usize, b: usize, t: usize, nh: usize, hs: usize) {
    debug_assert!(hs % 2 == 0, "RoPE needs an even head size");
    let c = nh * hs;
    let c3 = 3 * c;
    let half = hs / 2;
    let base = cast::<F>(10000.0);
    for bi in 0..b {
        for ti in 0..t {
            let pos = cast::<F>(ti as f64);
            let tok = qkv_off + (bi * t + ti) * c3;
            for h in 0..nh {
                let q = tok + h * hs;
                let k = tok + c + h * hs;
                for j in 0..half {
                    let inv_freq = base.powf(cast::<F>(-2.0 * j as f64 / hs as f64));
                    let theta = pos * inv_freq;
                    let (s, co) = (theta.sin(), theta.cos());
                    for off in [q, k] {
                        let x1 = acts[off + j];
                        let x2 = acts[off + j + half];
                        acts[off + j] = x1 * co - x2 * s;
                        acts[off + j + half] = x1 * s + x2 * co;
                    }
                }
            }
        }
    }
}

/// Backward of RoPE. The rotation is orthogonal, so the gradient is rotated by
/// the transpose (by `-angle`). Operates in place on the dq/dk grads in `gacts`.
fn rope_backward<F: Float>(gacts: &mut [F], qkv_off: usize, b: usize, t: usize, nh: usize, hs: usize) {
    let c = nh * hs;
    let c3 = 3 * c;
    let half = hs / 2;
    let base = cast::<F>(10000.0);
    for bi in 0..b {
        for ti in 0..t {
            let pos = cast::<F>(ti as f64);
            let tok = qkv_off + (bi * t + ti) * c3;
            for h in 0..nh {
                let q = tok + h * hs;
                let k = tok + c + h * hs;
                for j in 0..half {
                    let inv_freq = base.powf(cast::<F>(-2.0 * j as f64 / hs as f64));
                    let theta = pos * inv_freq;
                    let (s, co) = (theta.sin(), theta.cos());
                    for off in [q, k] {
                        let d1 = gacts[off + j];
                        let d2 = gacts[off + j + half];
                        gacts[off + j] = d1 * co + d2 * s;
                        gacts[off + j + half] = -d1 * s + d2 * co;
                    }
                }
            }
        }
    }
}

/// RMSNorm (Llama): `out = x · rsqrt(mean(x²)+eps) · weight`. No mean-centering,
/// no bias. Caches `rstd = rsqrt(mean(x²)+eps)` per row for the backward.
fn rmsnorm_forward<F: Float>(
    acts: &mut [F],
    out_off: usize,
    rstd_off: usize,
    inp_off: usize,
    weight: &[F],
    n: usize,
    c: usize,
) {
    let eps = cast::<F>(1e-5);
    let cf = cast::<F>(c as f64);
    for r in 0..n {
        let base = inp_off + r * c;
        let mut ms = F::zero();
        for i in 0..c {
            ms = ms + acts[base + i] * acts[base + i];
        }
        ms = ms / cf;
        let rstd = F::one() / (ms + eps).sqrt();
        let ob = out_off + r * c;
        for i in 0..c {
            acts[ob + i] = acts[base + i] * rstd * weight[i];
        }
        acts[rstd_off + r] = rstd;
    }
}

/// Split two disjoint sub-ranges out of one buffer as mutable slices, in either
/// order. Returns `(a, b)` for the requested `(a_off, a_len)` and `(b_off, b_len)`.
/// Lets parallel ops read one arena tensor while writing another.
fn disjoint_mut<F>(
    buf: &mut [F],
    a_off: usize,
    a_len: usize,
    b_off: usize,
    b_len: usize,
) -> (&mut [F], &mut [F]) {
    assert!(
        a_off + a_len <= b_off || b_off + b_len <= a_off,
        "ranges overlap"
    );
    if a_off < b_off {
        let (l, r) = buf.split_at_mut(b_off);
        (&mut l[a_off..a_off + a_len], &mut r[..b_len])
    } else {
        let (l, r) = buf.split_at_mut(a_off);
        (&mut r[..a_len], &mut l[b_off..b_off + b_len])
    }
}

/// Linear layer with weight `[OC, C]` (row = output unit): `out = inp @ W^T + b`.
/// Parallel over output rows; each output element's dot product stays in one
/// thread, so results are identical to the serial version.
fn matmul_forward<F: Float + Send + Sync>(
    out: &mut [F],
    inp: &[F],
    weight: &[F],
    bias: Option<&[F]>,
    n: usize,
    c: usize,
    oc: usize,
) {
    debug_assert_eq!(out.len(), n * oc);
    debug_assert_eq!(inp.len(), n * c);
    out.par_chunks_mut(oc).enumerate().for_each(|(r, orow)| {
        let irow = &inp[r * c..(r + 1) * c];
        for o in 0..oc {
            let mut val = bias.map_or(F::zero(), |b| b[o]);
            let w = &weight[o * c..(o + 1) * c];
            for i in 0..c {
                val = val + irow[i] * w[i];
            }
            orow[o] = val;
        }
    });
}

/// Arena adapter for `matmul_forward`: splits `inp`/`out` tensors out of `acts`.
fn linear<F: Float + Send + Sync>(
    acts: &mut [F],
    out_off: usize,
    inp_off: usize,
    weight: &[F],
    bias: Option<&[F]>,
    n: usize,
    c: usize,
    oc: usize,
) {
    let (inp, out) = disjoint_mut(acts, inp_off, n * c, out_off, n * oc);
    matmul_forward(out, inp, weight, bias, n, c, oc);
}

/// Apply (or undo, in backward) a precomputed dropout keep-mask in place: multiply
/// the `n` activations (or their gradients) at `x_off` by the mask at `m_off`. The
/// mask lives in a separate buffer, so the read and write never alias.
fn apply_dropout<F: Float + Send + Sync>(acts: &mut [F], x_off: usize, mask: &[F], m_off: usize, n: usize) {
    acts[x_off..x_off + n]
        .par_iter_mut()
        .zip(mask[m_off..m_off + n].par_iter())
        .for_each(|(x, &m)| *x = *x * m);
}

/// Causal multi-head self-attention. `inp` is the packed qkv `[B, T, 3C]`.
/// Parallel over the batch dimension: each `bi` writes disjoint regions of
/// `out`/`preatt`/`att` and only reads `inp`, so the math is identical to the
/// serial version (no reduction is reordered across threads).
fn attention_forward<F: Float + Send + Sync>(
    acts: &mut [F],
    out_off: usize,
    preatt_off: usize,
    att_off: usize,
    inp_off: usize,
    b: usize,
    t: usize,
    c: usize,
    nh: usize,
    att_mask: Option<&[F]>,
) {
    let hs = c / nh;
    let scale = F::one() / cast::<F>(hs as f64).sqrt();
    let c3 = 3 * c;
    // Arena order is inp(qkv) < out(atty) < preatt < att; carve out the four
    // disjoint tensors so the batch loop can run in parallel.
    let (inp, out, preatt, att) = four_disjoint_mut(
        acts,
        inp_off, b * t * c3,
        out_off, b * t * c,
        preatt_off, b * nh * t * t,
        att_off, b * nh * t * t,
    );
    let inp: &[F] = inp;
    out.par_chunks_mut(t * c)
        .zip(preatt.par_chunks_mut(nh * t * t))
        .zip(att.par_chunks_mut(nh * t * t))
        .zip(inp.par_chunks(t * c3))
        .enumerate()
        .for_each(|(bi, (((out_b, pre_b), att_b), inp_b))| {
            let mask_base = bi * nh * t * t; // into att_mask, if any
            for h in 0..nh {
                for ti in 0..t {
                    let q_base = ti * c3 + h * hs;
                    let pre_base = (h * t + ti) * t;
                    let att_base = pre_base;

                    // scores against keys at positions <= ti, tracking the max
                    let mut maxval = F::neg_infinity();
                    for t2 in 0..=ti {
                        let k_base = t2 * c3 + c + h * hs;
                        let mut val = F::zero();
                        for i in 0..hs {
                            val = val + inp_b[q_base + i] * inp_b[k_base + i];
                        }
                        val = val * scale;
                        pre_b[pre_base + t2] = val;
                        if val > maxval {
                            maxval = val;
                        }
                    }
                    // softmax over the causal window
                    let mut sum = F::zero();
                    for t2 in 0..=ti {
                        let e = (pre_b[pre_base + t2] - maxval).exp();
                        att_b[att_base + t2] = e;
                        sum = sum + e;
                    }
                    let inv = if sum > F::zero() { F::one() / sum } else { F::zero() };
                    for t2 in 0..t {
                        if t2 <= ti {
                            att_b[att_base + t2] = att_b[att_base + t2] * inv;
                        } else {
                            // upper triangle is masked out
                            att_b[att_base + t2] = F::zero();
                            pre_b[pre_base + t2] = F::zero();
                        }
                    }
                    // weighted sum of values
                    let o_base = ti * c + h * hs;
                    for i in 0..hs {
                        out_b[o_base + i] = F::zero();
                    }
                    for t2 in 0..=ti {
                        let v_base = t2 * c3 + 2 * c + h * hs;
                        // dropout on the (cached pre-dropout) attention weight
                        let a = att_mask.map_or(att_b[att_base + t2], |m| {
                            att_b[att_base + t2] * m[mask_base + att_base + t2]
                        });
                        for i in 0..hs {
                            out_b[o_base + i] = out_b[o_base + i] + a * inp_b[v_base + i];
                        }
                    }
                }
            }
        });
}

/// Split four mutually disjoint sub-slices out of one buffer, given in ascending
/// offset order (each `[off, off+len)` must not overlap the next).
fn four_disjoint_mut<F>(
    buf: &mut [F],
    o0: usize, l0: usize,
    o1: usize, l1: usize,
    o2: usize, l2: usize,
    o3: usize, l3: usize,
) -> (&mut [F], &mut [F], &mut [F], &mut [F]) {
    assert!(o0 + l0 <= o1 && o1 + l1 <= o2 && o2 + l2 <= o3, "ranges overlap or out of order");
    let (_, rest) = buf.split_at_mut(o0);
    let (s0, rest) = rest.split_at_mut(l0);
    let (_, rest) = rest.split_at_mut(o1 - (o0 + l0));
    let (s1, rest) = rest.split_at_mut(l1);
    let (_, rest) = rest.split_at_mut(o2 - (o1 + l1));
    let (s2, rest) = rest.split_at_mut(l2);
    let (_, rest) = rest.split_at_mut(o3 - (o2 + l2));
    let (s3, _) = rest.split_at_mut(l3);
    (s0, s1, s2, s3)
}

fn residual_forward<F: Float>(acts: &mut [F], out_off: usize, in1_off: usize, in2_off: usize, n: usize) {
    for i in 0..n {
        acts[out_off + i] = acts[in1_off + i] + acts[in2_off + i];
    }
}

/// Row-wise softmax of `logits [n, v]` into `probs [n, v]` (numerically stable).
fn softmax_forward<F: Float + Send + Sync>(acts: &mut [F], probs_off: usize, logits_off: usize, n: usize, v: usize) {
    // arena order: logits < probs
    let (logits, probs) = disjoint_mut(acts, logits_off, n * v, probs_off, n * v);
    probs
        .par_chunks_mut(v)
        .zip(logits.par_chunks(v))
        .for_each(|(prow, lrow)| {
            let mut maxval = F::neg_infinity();
            for &l in lrow.iter() {
                if l > maxval {
                    maxval = l;
                }
            }
            let mut sum = F::zero();
            for i in 0..v {
                let e = (lrow[i] - maxval).exp();
                prow[i] = e;
                sum = sum + e;
            }
            let inv = F::one() / sum;
            for p in prow.iter_mut() {
                *p = *p * inv;
            }
        });
}

/// Per-token cross-entropy `losses[r] = -ln(probs[r, target[r]])`.
fn crossentropy_forward<F: Float>(
    acts: &mut [F],
    losses_off: usize,
    probs_off: usize,
    targets: &[u16],
    n: usize,
    v: usize,
) {
    for r in 0..n {
        let ix = targets[r] as usize;
        acts[losses_off + r] = -acts[probs_off + r * v + ix].ln();
    }
}

/// SwiGLU gate: `glu = silu(gate) · up`, with `silu(z) = z·σ(z)`. `gate`, `up`,
/// `glu` are distinct arena tensors (gate read, up read, glu written).
fn swiglu_forward<F: Float>(acts: &mut [F], glu_off: usize, gate_off: usize, up_off: usize, n: usize) {
    for i in 0..n {
        let g = acts[gate_off + i];
        let u = acts[up_off + i];
        let silu = g / (F::one() + (-g).exp());
        acts[glu_off + i] = silu * u;
    }
}

// ---------------------------------------------------------------------------
// Backward ops, generic over the float type (same code runs in f32 for
// training and f64 for the gradient check). Each accumulates (+=) into its
// gradient targets, so the branching residual stream sums correctly; all grad
// buffers are zeroed first. `gacts` = activation grads (arena layout),
// `grads` = param grads (param layout), `acts`/`params` = cached forward values.
// ---------------------------------------------------------------------------

/// Fused softmax + cross-entropy: `dlogits[r,j] += (probs[r,j] - 1{j=t}) * dmean`.
fn crossentropy_softmax_backward<F: Float + Send + Sync>(
    gacts: &mut [F],
    dlogits_off: usize,
    acts: &[F],
    probs_off: usize,
    targets: &[u16],
    n: usize,
    v: usize,
    dmean: F,
) {
    let dlogits = &mut gacts[dlogits_off..dlogits_off + n * v];
    let probs = &acts[probs_off..probs_off + n * v];
    dlogits
        .par_chunks_mut(v)
        .zip(probs.par_chunks(v))
        .enumerate()
        .for_each(|(r, (drow, prow))| {
            let ix = targets[r] as usize;
            for j in 0..v {
                let ind = if j == ix { F::one() } else { F::zero() };
                drow[j] = drow[j] + (prow[j] - ind) * dmean;
            }
        });
}

/// Backward of the `[OC, C]` linear. Accumulates into `dinp`, `dweight`, `dbias`
/// (each fresh/zeroed). `dinp` is parallel over rows, the weight/bias reduction
/// is parallel over output units, so per-element accumulation order matches the
/// serial version.
fn matmul_backward<F: Float + Send + Sync>(
    dinp: &mut [F],
    dout: &[F],
    dweight: &mut [F],
    dbias: Option<&mut [F]>,
    inp: &[F],
    weight: &[F],
    n: usize,
    c: usize,
    oc: usize,
) {
    // dinp[r,:] += sum_o dout[r,o] * weight[o,:]
    dinp.par_chunks_mut(c).enumerate().for_each(|(r, dinp_r)| {
        let dout_r = &dout[r * oc..(r + 1) * oc];
        for o in 0..oc {
            let d = dout_r[o];
            let w = &weight[o * c..(o + 1) * c];
            for i in 0..c {
                dinp_r[i] = dinp_r[i] + w[i] * d;
            }
        }
    });
    // dweight[o,:] += sum_r dout[r,o] * inp[r,:] ; dbias[o] += sum_r dout[r,o]
    match dbias {
        Some(dbias) => {
            dweight
                .par_chunks_mut(c)
                .zip(dbias.par_iter_mut())
                .enumerate()
                .for_each(|(o, (dw, db))| {
                    let mut acc = F::zero();
                    for r in 0..n {
                        let d = dout[r * oc + o];
                        let inp_r = &inp[r * c..(r + 1) * c];
                        for i in 0..c {
                            dw[i] = dw[i] + inp_r[i] * d;
                        }
                        acc = acc + d;
                    }
                    *db = *db + acc;
                });
        }
        None => {
            dweight.par_chunks_mut(c).enumerate().for_each(|(o, dw)| {
                for r in 0..n {
                    let d = dout[r * oc + o];
                    let inp_r = &inp[r * c..(r + 1) * c];
                    for i in 0..c {
                        dw[i] = dw[i] + inp_r[i] * d;
                    }
                }
            });
        }
    }
}

/// Arena adapter for `matmul_backward`: carves the `dinp`/`dout` tensors out of
/// `gacts` and the `dweight`/`dbias` tensors out of `grads`.
fn linear_backward<F: Float + Send + Sync>(
    gacts: &mut [F],
    dinp_off: usize,
    dout_off: usize,
    grads: &mut [F],
    dweight_off: usize,
    dbias_off: Option<usize>,
    acts: &[F],
    inp_off: usize,
    params: &[F],
    weight_off: usize,
    n: usize,
    c: usize,
    oc: usize,
) {
    let (dinp, dout) = disjoint_mut(gacts, dinp_off, n * c, dout_off, n * oc);
    let inp = &acts[inp_off..inp_off + n * c];
    let weight = &params[weight_off..weight_off + oc * c];
    match dbias_off {
        Some(dbo) => {
            let (dweight, dbias) = disjoint_mut(grads, dweight_off, oc * c, dbo, oc);
            matmul_backward(dinp, dout, dweight, Some(dbias), inp, weight, n, c, oc);
        }
        None => {
            let dweight = &mut grads[dweight_off..dweight_off + oc * c];
            matmul_backward(dinp, dout, dweight, None, inp, weight, n, c, oc);
        }
    }
}

/// Backward of LayerNorm. Accumulates `dinp`, `dweight`, `dbias`.
/// Backward of RMSNorm. Like LayerNorm's but without the mean-centering term:
/// `dx_k = rstd·(dnorm_k − norm_k·⟨dnorm·norm⟩)`, with `norm = x·rstd`,
/// `dnorm = weight·dout`. Accumulates `dinp` and `dweight` (no bias).
fn rmsnorm_backward<F: Float>(
    gacts: &mut [F],
    dinp_off: usize,
    dout_off: usize,
    grads: &mut [F],
    dweight_off: usize,
    acts: &[F],
    inp_off: usize,
    rstd_off: usize,
    params: &[F],
    weight_off: usize,
    n: usize,
    c: usize,
) {
    let cf = cast::<F>(c as f64);
    for r in 0..n {
        let rstd = acts[rstd_off + r];
        let ib = inp_off + r * c;
        let ob = dout_off + r * c;
        let dib = dinp_off + r * c;

        // single reduction term over the row
        let mut dnorm_norm_mean = F::zero();
        for i in 0..c {
            let norm = acts[ib + i] * rstd;
            let dnorm_i = params[weight_off + i] * gacts[ob + i];
            dnorm_norm_mean = dnorm_norm_mean + dnorm_i * norm;
        }
        dnorm_norm_mean = dnorm_norm_mean / cf;

        for i in 0..c {
            let norm = acts[ib + i] * rstd;
            let dnorm_i = params[weight_off + i] * gacts[ob + i];
            grads[dweight_off + i] = grads[dweight_off + i] + norm * gacts[ob + i];
            let dval = (dnorm_i - norm * dnorm_norm_mean) * rstd;
            gacts[dib + i] = gacts[dib + i] + dval;
        }
    }
}

/// Backward of SwiGLU. From `dglu` (grad of `glu = silu(gate)·up`) produce
/// `dgate` and `dup`. `silu'(g) = σ(g)·(1 + g·(1−σ(g)))`. Writes dgate/dup
/// (each produced only here).
fn swiglu_backward<F: Float>(
    gacts: &mut [F],
    dgate_off: usize,
    dup_off: usize,
    dglu_off: usize,
    acts: &[F],
    gate_off: usize,
    up_off: usize,
    n: usize,
) {
    for i in 0..n {
        let g = acts[gate_off + i];
        let u = acts[up_off + i];
        let sig = F::one() / (F::one() + (-g).exp());
        let silu = g * sig;
        let dsilu = sig * (F::one() + g * (F::one() - sig));
        let dglu = gacts[dglu_off + i];
        gacts[dgate_off + i] = dglu * u * dsilu;
        gacts[dup_off + i] = dglu * silu;
    }
}

/// Backward of a residual add: both inputs get the upstream gradient.
fn residual_backward<F: Float>(gacts: &mut [F], dinp1_off: usize, dinp2_off: usize, dout_off: usize, n: usize) {
    for i in 0..n {
        let d = gacts[dout_off + i];
        gacts[dinp1_off + i] = gacts[dinp1_off + i] + d;
        gacts[dinp2_off + i] = gacts[dinp2_off + i] + d;
    }
}

/// Backward of causal multi-head attention. Accumulates `dinp` (dqkv), using
/// `dpreatt`/`datt` as scratch. `inp`/`att` are the cached forward values.
fn attention_backward<F: Float + Send + Sync>(
    gacts: &mut [F],
    dinp_off: usize,
    dpreatt_off: usize,
    datt_off: usize,
    dout_off: usize,
    acts: &[F],
    inp_off: usize,
    att_off: usize,
    b: usize,
    t: usize,
    c: usize,
    nh: usize,
    att_mask: Option<&[F]>,
) {
    let hs = c / nh;
    let scale = F::one() / cast::<F>(hs as f64).sqrt();
    let c3 = 3 * c;
    // Same batch-parallel split as the forward; `acts` is read-only/shared.
    // Gradient-arena order mirrors the forward arena: dinp(qkv) < dout(atty) <
    // dpreatt < datt.
    let (dinp, dout, dpreatt, datt) = four_disjoint_mut(
        gacts,
        dinp_off, b * t * c3,
        dout_off, b * t * c,
        dpreatt_off, b * nh * t * t,
        datt_off, b * nh * t * t,
    );
    dinp.par_chunks_mut(t * c3)
        .zip(dout.par_chunks_mut(t * c))
        .zip(dpreatt.par_chunks_mut(nh * t * t))
        .zip(datt.par_chunks_mut(nh * t * t))
        .enumerate()
        .for_each(|(bi, (((dinp_b, dout_b), dpreatt_b), datt_b))| {
            let att_bi = att_off + bi * nh * t * t;
            let inp_bi = inp_off + bi * t * c3;
            let mask_bi = bi * nh * t * t; // into att_mask, if any
            for h in 0..nh {
                for ti in 0..t {
                    let att_base = att_bi + (h * t + ti) * t;
                    let datt_base = (h * t + ti) * t;
                    let dpreatt_base = datt_base;
                    let dout_base = ti * c + h * hs;

                    // backward through value accumulation + attn-weight dropout:
                    // dvalue uses the post-dropout weight; datt becomes d(att_sm).
                    for t2 in 0..=ti {
                        let v_base = inp_bi + t2 * c3 + 2 * c + h * hs;
                        let dv_base = t2 * c3 + 2 * c + h * hs;
                        let m2 = att_mask.map_or(F::one(), |m| m[mask_bi + datt_base + t2]);
                        let att_do = acts[att_base + t2] * m2; // post-dropout weight
                        for i in 0..hs {
                            datt_b[datt_base + t2] =
                                datt_b[datt_base + t2] + acts[v_base + i] * dout_b[dout_base + i];
                            dinp_b[dv_base + i] =
                                dinp_b[dv_base + i] + att_do * dout_b[dout_base + i];
                        }
                        // datt held d(att_do); fold the dropout mask -> d(att_sm)
                        datt_b[datt_base + t2] = datt_b[datt_base + t2] * m2;
                    }
                    // backward through the softmax: dpreatt = (diag(att) - att att^T) datt
                    for t2 in 0..=ti {
                        for t3 in 0..=ti {
                            let ind = if t2 == t3 { F::one() } else { F::zero() };
                            let local = acts[att_base + t2] * (ind - acts[att_base + t3]);
                            dpreatt_b[dpreatt_base + t3] =
                                dpreatt_b[dpreatt_base + t3] + local * datt_b[datt_base + t2];
                        }
                    }
                    // backward through the scaled dot product -> dquery, dkey
                    for t2 in 0..=ti {
                        let q_base = inp_bi + ti * c3 + h * hs;
                        let dq_base = ti * c3 + h * hs;
                        let k_base = inp_bi + t2 * c3 + c + h * hs;
                        let dk_base = t2 * c3 + c + h * hs;
                        let dpre = dpreatt_b[dpreatt_base + t2] * scale;
                        for i in 0..hs {
                            dinp_b[dq_base + i] = dinp_b[dq_base + i] + acts[k_base + i] * dpre;
                            dinp_b[dk_base + i] = dinp_b[dk_base + i] + acts[q_base + i] * dpre;
                        }
                    }
                }
            }
        });
}

/// Backward of the encoder: scatter-add into `dwte` (by token id) and `dwpe`.
fn encoder_backward<F: Float>(
    grads: &mut [F],
    dwte_off: usize,
    gacts: &[F],
    dout_off: usize,
    ids: &[u16],
    b: usize,
    t: usize,
    c: usize,
) {
    for bi in 0..b {
        for ti in 0..t {
            let ix = ids[bi * t + ti] as usize;
            let o = dout_off + (bi * t + ti) * c;
            for ci in 0..c {
                grads[dwte_off + ix * c + ci] = grads[dwte_off + ix * c + ci] + gacts[o + ci];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn forward_shape_and_finite() {
        let cfg = Config {
            n_layer: 2,
            n_head: 2,
            n_embd: 16,
            block_size: 4,
            vocab_size: 8,
            batch_size: 2,
        };
        let mut rng = StdRng::seed_from_u64(0);
        let mut model = Gpt::new(cfg, &mut rng);
        let ids: Vec<u16> = (0..cfg.batch_size * cfg.block_size)
            .map(|i| (i % cfg.vocab_size) as u16)
            .collect();
        model.forward(&ids, None);

        let logits = model.logits();
        assert_eq!(logits.len(), cfg.batch_size * cfg.block_size * cfg.vocab_size);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn untrained_loss_near_ln_vocab() {
        let cfg = Config {
            n_layer: 2,
            n_head: 4,
            n_embd: 32,
            block_size: 16,
            vocab_size: 128,
            batch_size: 4,
        };
        let mut rng = StdRng::seed_from_u64(42);
        let mut model = Gpt::new(cfg, &mut rng);
        let n = cfg.batch_size * cfg.block_size;
        let ids: Vec<u16> = (0..n).map(|i| (i % cfg.vocab_size) as u16).collect();
        let targets: Vec<u16> = (0..n).map(|i| ((i * 7) % cfg.vocab_size) as u16).collect();

        let loss = model.forward(&ids, Some(&targets)).unwrap();
        let expected = (cfg.vocab_size as f32).ln();
        assert!(
            (loss - expected).abs() < 0.5,
            "untrained loss {loss} far from ln(vocab) {expected}"
        );
    }

    #[test]
    fn adamw_overfits_single_batch() {
        let cfg = Config {
            n_layer: 2,
            n_head: 2,
            n_embd: 16,
            block_size: 8,
            vocab_size: 32,
            batch_size: 2,
        };
        let mut rng = StdRng::seed_from_u64(7);
        let mut model = Gpt::new(cfg, &mut rng);
        let n = cfg.batch_size * cfg.block_size;
        let ids: Vec<u16> = (0..n).map(|i| ((i * 5 + 3) % cfg.vocab_size) as u16).collect();
        let targets: Vec<u16> = (0..n).map(|i| ((i * 11 + 1) % cfg.vocab_size) as u16).collect();

        let first = model.forward(&ids, Some(&targets)).unwrap();
        let mut last = first;
        for _ in 0..300 {
            last = model.forward(&ids, Some(&targets)).unwrap();
            model.backward(&ids, &targets);
            model.adamw_step(1e-2, 0.9, 0.999, 1e-8, 0.0);
        }
        assert!(
            last < first * 0.1,
            "loss did not drop enough: {first} -> {last}"
        );
    }

    /// Gradient-check the dropout path with a *fixed* f64 mask (so the forward is
    /// deterministic and finite differences are valid). Exercises all four dropout
    /// sites including attention-weight dropout.
    #[test]
    fn dropout_backward_gradient_check() {
        let cfg = Config {
            n_layer: 2, n_head: 2, n_embd: 16, block_size: 6, vocab_size: 32, batch_size: 2,
        };
        let mut rng = StdRng::seed_from_u64(123);
        let model = Gpt::new(cfg, &mut rng);
        let pl = ParamLayout::new(&cfg);
        let al = ActLayout::new(&cfg);
        let ml = MaskLayout::new(&cfg);
        let n = cfg.batch_size * cfg.block_size;
        let ids: Vec<u16> = (0..n).map(|i| (i % cfg.vocab_size) as u16).collect();
        let targets: Vec<u16> = (0..n).map(|i| ((i * 7 + 1) % cfg.vocab_size) as u16).collect();

        let params: Vec<f64> = (0..pl.total).map(|i| model.param(i) as f64).collect();
        // fixed keep-mask: ~1 in 4 dropped, the rest scaled by 1/keep
        let keep = 0.75f64;
        let scale = 1.0 / keep;
        let mask: Vec<f64> = (0..ml.total)
            .map(|i| if i.wrapping_mul(2_654_435_761) % 4 == 0 { 0.0 } else { scale })
            .collect();

        let mut acts = vec![0.0f64; al.total];
        forward_into::<f64>(&cfg, &params, &mut acts, &ids, Some(&targets), Some(&mask));
        let mut grads = vec![0.0f64; pl.total];
        let mut gacts = vec![0.0f64; al.total];
        backward_into::<f64>(&cfg, &params, &acts, &mut grads, &mut gacts, &ids, &targets, Some(&mask));

        let loss = |p: &[f64]| {
            let mut a = vec![0.0f64; al.total];
            forward_into::<f64>(&cfg, p, &mut a, &ids, Some(&targets), Some(&mask)).unwrap()
        };
        let eps = 1e-6;
        let mut worst = 0.0f64;
        for &i in &[
            pl.wte.off, pl.qkvw.off, pl.attprojw.off, pl.w1.off, pl.w2.off,
            pl.ln1w.off, pl.ln2w.off, pl.lm_head.off, pl.lnfw.off,
        ] {
            let mut pp = params.clone();
            pp[i] = params[i] + eps;
            let lp = loss(&pp);
            pp[i] = params[i] - eps;
            let lm = loss(&pp);
            let num = (lp - lm) / (2.0 * eps);
            let ana = grads[i];
            let rel = (num - ana).abs() / num.abs().max(ana.abs()).max(1e-9);
            worst = worst.max(rel);
        }
        assert!(worst < 1e-5, "dropout gradient check failed: {worst:e}");
    }
}
