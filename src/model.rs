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
    wpe: Off,      // (maxT, C) positional embedding
    ln1w: Off,     // (L, C)
    ln1b: Off,     // (L, C)
    qkvw: Off,     // (L, 3C, C)
    qkvb: Off,     // (L, 3C)
    attprojw: Off, // (L, C, C)
    attprojb: Off, // (L, C)
    ln2w: Off,     // (L, C)
    ln2b: Off,     // (L, C)
    fcw: Off,      // (L, 4C, C)
    fcb: Off,      // (L, 4C)
    fcprojw: Off,  // (L, C, 4C)
    fcprojb: Off,  // (L, C)
    lnfw: Off,     // (C)
    lnfb: Off,     // (C)
    total: usize,
}

impl ParamLayout {
    fn new(c: &Config) -> Self {
        let (l, v, ch, mt) = (c.n_layer, c.vocab_size, c.n_embd, c.block_size);
        let mut o = 0usize;
        let mut take = |n: usize| {
            let off = o;
            o += n;
            Off { off, len: n }
        };
        let wte = take(v * ch);
        let wpe = take(mt * ch);
        let ln1w = take(l * ch);
        let ln1b = take(l * ch);
        let qkvw = take(l * 3 * ch * ch);
        let qkvb = take(l * 3 * ch);
        let attprojw = take(l * ch * ch);
        let attprojb = take(l * ch);
        let ln2w = take(l * ch);
        let ln2b = take(l * ch);
        let fcw = take(l * 4 * ch * ch);
        let fcb = take(l * 4 * ch);
        let fcprojw = take(l * ch * 4 * ch);
        let fcprojb = take(l * ch);
        let lnfw = take(ch);
        let lnfb = take(ch);
        Self {
            wte, wpe, ln1w, ln1b, qkvw, qkvb, attprojw, attprojb, ln2w, ln2b,
            fcw, fcb, fcprojw, fcprojb, lnfw, lnfb, total: o,
        }
    }
}

/// Offsets of every activation tensor inside the flat `acts` arena (sized B,T).
#[derive(Clone, Copy)]
struct ActLayout {
    encoded: Off,   // (B, T, C)
    ln1: Off,       // (L, B, T, C)
    ln1_mean: Off,  // (L, B, T)
    ln1_rstd: Off,  // (L, B, T)
    qkv: Off,       // (L, B, T, 3C)
    atty: Off,      // (L, B, T, C)
    preatt: Off,    // (L, B, NH, T, T)
    att: Off,       // (L, B, NH, T, T)
    attproj: Off,   // (L, B, T, C)
    residual2: Off, // (L, B, T, C)
    ln2: Off,       // (L, B, T, C)
    ln2_mean: Off,  // (L, B, T)
    ln2_rstd: Off,  // (L, B, T)
    fch: Off,       // (L, B, T, 4C)
    fch_gelu: Off,  // (L, B, T, 4C)
    fcproj: Off,    // (L, B, T, C)
    residual3: Off, // (L, B, T, C)
    lnf: Off,       // (B, T, C)
    lnf_mean: Off,  // (B, T)
    lnf_rstd: Off,  // (B, T)
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
        let ln1_mean = take(l * bt);
        let ln1_rstd = take(l * bt);
        let qkv = take(l * bt * 3 * ch);
        let atty = take(l * bt * ch);
        let preatt = take(l * b * nh * t * t);
        let att = take(l * b * nh * t * t);
        let attproj = take(l * bt * ch);
        let residual2 = take(l * bt * ch);
        let ln2 = take(l * bt * ch);
        let ln2_mean = take(l * bt);
        let ln2_rstd = take(l * bt);
        let fch = take(l * bt * 4 * ch);
        let fch_gelu = take(l * bt * 4 * ch);
        let fcproj = take(l * bt * ch);
        let residual3 = take(l * bt * ch);
        let lnf = take(bt * ch);
        let lnf_mean = take(bt);
        let lnf_rstd = take(bt);
        let logits = take(bt * v);
        let probs = take(bt * v);
        let losses = take(bt);
        Self {
            encoded, ln1, ln1_mean, ln1_rstd, qkv, atty, preatt, att, attproj,
            residual2, ln2, ln2_mean, ln2_rstd, fch, fch_gelu, fcproj, residual3,
            lnf, lnf_mean, lnf_rstd, logits, probs, losses, total: o,
        }
    }
}

/// A GPT-2-style transformer: flat params + activation arena, with matching
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
    pl: ParamLayout,
    al: ActLayout,
}

impl Gpt {
    /// Allocate and randomly initialise a model for a fixed `(batch_size, block_size)`.
    pub fn new<R: Rng>(cfg: Config, rng: &mut R) -> Self {
        assert!(cfg.n_embd % cfg.n_head == 0, "n_embd must divide by n_head");
        let pl = ParamLayout::new(&cfg);
        let al = ActLayout::new(&cfg);
        let mut params = vec![0.0f32; pl.total];

        // Weights ~ N(0, 0.02); biases 0; LayerNorm weights 1.
        for o in [pl.wte, pl.wpe, pl.qkvw, pl.attprojw, pl.fcw, pl.fcprojw] {
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
        Self { cfg, params, acts, grads, gacts, m, v, adam_t: 0, pl, al }
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
        forward_into::<f32>(&self.cfg, &self.params, &mut self.acts, ids, targets)
    }

    /// Recompute only the scalar loss in f64 (params cast to f64, fresh f64
    /// activation scratch). The gradient-check harness uses this so numerical
    /// gradients aren't limited by f32 round-off (~1e-3) and can hit < 1e-4.
    pub fn loss_f64(&self, ids: &[u16], targets: &[u16]) -> f64 {
        let params: Vec<f64> = self.params.iter().map(|&x| x as f64).collect();
        let mut acts = vec![0.0f64; self.al.total];
        forward_into::<f64>(&self.cfg, &params, &mut acts, ids, Some(targets)).unwrap()
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
            ("wpe", p.wpe.off, p.wpe.len),
            ("ln1w", p.ln1w.off, p.ln1w.len),
            ("ln1b", p.ln1b.off, p.ln1b.len),
            ("qkvw", p.qkvw.off, p.qkvw.len),
            ("qkvb", p.qkvb.off, p.qkvb.len),
            ("attprojw", p.attprojw.off, p.attprojw.len),
            ("attprojb", p.attprojb.off, p.attprojb.len),
            ("ln2w", p.ln2w.off, p.ln2w.len),
            ("ln2b", p.ln2b.off, p.ln2b.len),
            ("fcw", p.fcw.off, p.fcw.len),
            ("fcb", p.fcb.off, p.fcb.len),
            ("fcprojw", p.fcprojw.off, p.fcprojw.len),
            ("fcprojb", p.fcprojb.off, p.fcprojb.len),
            ("lnfw", p.lnfw.off, p.lnfw.len),
            ("lnfb", p.lnfb.off, p.lnfb.len),
        ]
    }

    /// Backward pass. Requires a prior `forward(ids, Some(targets))`; fills the
    /// f32 `grads` (param) and `gacts` (activation) buffers from zero.
    pub fn backward(&mut self, ids: &[u16], targets: &[u16]) {
        backward_into::<f32>(
            &self.cfg, &self.params, &self.acts, &mut self.grads, &mut self.gacts, ids, targets,
        );
    }

    /// Recompute the parameter gradients entirely in f64 (fresh f64 forward +
    /// backward). The gradient checker compares these against f64 numerical
    /// gradients, isolating formula correctness from f32 round-off so the match
    /// lands well under 1e-4.
    pub fn grads_f64(&self, ids: &[u16], targets: &[u16]) -> Vec<f64> {
        let params: Vec<f64> = self.params.iter().map(|&x| x as f64).collect();
        let mut acts = vec![0.0f64; self.al.total];
        forward_into::<f64>(&self.cfg, &params, &mut acts, ids, Some(targets));
        let mut grads = vec![0.0f64; self.pl.total];
        let mut gacts = vec![0.0f64; self.al.total];
        backward_into::<f64>(&self.cfg, &params, &acts, &mut grads, &mut gacts, ids, targets);
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
) {
    let (b, t, ch, nh, v, nl) = (
        cfg.batch_size, cfg.block_size, cfg.n_embd, cfg.n_head, cfg.vocab_size, cfg.n_layer,
    );
    let bt = b * t;
    let pl = ParamLayout::new(cfg);
    let al = ActLayout::new(cfg);

    for x in grads.iter_mut() {
        *x = F::zero();
    }
    for x in gacts.iter_mut() {
        *x = F::zero();
    }

    // loss = mean over B*T tokens -> d loss / d losses[r] = 1/(B*T)
    let dmean = F::one() / cast::<F>(bt as f64);

    crossentropy_softmax_backward(gacts, al.logits.off, acts, al.probs.off, targets, bt, v, dmean);
    // lm_head (logits = lnf @ wte^T, tied weight, no bias)
    linear_backward(
        gacts, al.lnf.off, al.logits.off,
        grads, pl.wte.off, None,
        acts, al.lnf.off, params, pl.wte.off, bt, ch, v,
    );
    let resf_off = al.residual3.off + (nl - 1) * bt * ch;
    layernorm_backward(
        gacts, resf_off, al.lnf.off,
        grads, pl.lnfw.off, pl.lnfb.off,
        acts, resf_off, al.lnf_mean.off, al.lnf_rstd.off,
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
        let ln1_mean_off = al.ln1_mean.off + l * bt;
        let ln1_rstd_off = al.ln1_rstd.off + l * bt;
        let qkv_off = al.qkv.off + l * bt * 3 * ch;
        let atty_off = al.atty.off + l * bt * ch;
        let preatt_off = al.preatt.off + l * b * nh * t * t;
        let att_off = al.att.off + l * b * nh * t * t;
        let attproj_off = al.attproj.off + l * bt * ch;
        let residual2_off = al.residual2.off + l * bt * ch;
        let ln2_off = al.ln2.off + l * bt * ch;
        let ln2_mean_off = al.ln2_mean.off + l * bt;
        let ln2_rstd_off = al.ln2_rstd.off + l * bt;
        let fch_off = al.fch.off + l * bt * 4 * ch;
        let fch_gelu_off = al.fch_gelu.off + l * bt * 4 * ch;
        let fcproj_off = al.fcproj.off + l * bt * ch;
        let residual3_off = al.residual3.off + l * bt * ch;

        // parameter (and param-grad) offsets
        let ln1w_off = pl.ln1w.off + l * ch;
        let ln1b_off = pl.ln1b.off + l * ch;
        let qkvw_off = pl.qkvw.off + l * 3 * ch * ch;
        let qkvb_off = pl.qkvb.off + l * 3 * ch;
        let attprojw_off = pl.attprojw.off + l * ch * ch;
        let attprojb_off = pl.attprojb.off + l * ch;
        let ln2w_off = pl.ln2w.off + l * ch;
        let ln2b_off = pl.ln2b.off + l * ch;
        let fcw_off = pl.fcw.off + l * 4 * ch * ch;
        let fcb_off = pl.fcb.off + l * 4 * ch;
        let fcprojw_off = pl.fcprojw.off + l * ch * 4 * ch;
        let fcprojb_off = pl.fcprojb.off + l * ch;

        // residual3 = residual2 + fcproj
        residual_backward(gacts, residual2_off, fcproj_off, residual3_off, bt * ch);
        // fcproj = fch_gelu @ fcprojw^T + fcprojb
        linear_backward(
            gacts, fch_gelu_off, fcproj_off,
            grads, fcprojw_off, Some(fcprojb_off),
            acts, fch_gelu_off, params, fcprojw_off, bt, 4 * ch, ch,
        );
        gelu_backward(gacts, fch_off, fch_gelu_off, acts, fch_off, bt * 4 * ch);
        // fch = ln2 @ fcw^T + fcb
        linear_backward(
            gacts, ln2_off, fch_off,
            grads, fcw_off, Some(fcb_off),
            acts, ln2_off, params, fcw_off, bt, ch, 4 * ch,
        );
        // ln2 = layernorm(residual2)  -> accumulates into d residual2
        layernorm_backward(
            gacts, residual2_off, ln2_off,
            grads, ln2w_off, ln2b_off,
            acts, residual2_off, ln2_mean_off, ln2_rstd_off,
            params, ln2w_off, bt, ch,
        );
        // residual2 = residual + attproj
        residual_backward(gacts, res_off, attproj_off, residual2_off, bt * ch);
        // attproj = atty @ attprojw^T + attprojb
        linear_backward(
            gacts, atty_off, attproj_off,
            grads, attprojw_off, Some(attprojb_off),
            acts, atty_off, params, attprojw_off, bt, ch, ch,
        );
        attention_backward(
            gacts, qkv_off, preatt_off, att_off, atty_off,
            acts, qkv_off, att_off, b, t, ch, nh,
        );
        // qkv = ln1 @ qkvw^T + qkvb
        linear_backward(
            gacts, ln1_off, qkv_off,
            grads, qkvw_off, Some(qkvb_off),
            acts, ln1_off, params, qkvw_off, bt, ch, 3 * ch,
        );
        // ln1 = layernorm(residual)  -> accumulates into d residual (layer input)
        layernorm_backward(
            gacts, res_off, ln1_off,
            grads, ln1w_off, ln1b_off,
            acts, res_off, ln1_mean_off, ln1_rstd_off,
            params, ln1w_off, bt, ch,
        );
    }

    // encoded = wte[ids] + wpe[pos]  (wte grad accumulates on top of lm_head)
    encoder_backward(grads, pl.wte.off, pl.wpe.off, gacts, al.encoded.off, ids, b, t, ch);
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
) -> Option<F> {
    let (b, t, ch, nh, v, nl) = (
        cfg.batch_size, cfg.block_size, cfg.n_embd, cfg.n_head, cfg.vocab_size, cfg.n_layer,
    );
    let bt = b * t;
    assert_eq!(ids.len(), bt, "ids must be batch_size * block_size");

    let pl = ParamLayout::new(cfg);
    let al = ActLayout::new(cfg);

    encoder_forward(
        acts, al.encoded.off, ids,
        &params[pl.wte.range()], &params[pl.wpe.range()], b, t, ch,
    );

    for l in 0..nl {
        let res_off = if l == 0 {
            al.encoded.off
        } else {
            al.residual3.off + (l - 1) * bt * ch
        };

        // Per-layer activation offsets.
        let ln1_off = al.ln1.off + l * bt * ch;
        let ln1_mean_off = al.ln1_mean.off + l * bt;
        let ln1_rstd_off = al.ln1_rstd.off + l * bt;
        let qkv_off = al.qkv.off + l * bt * 3 * ch;
        let atty_off = al.atty.off + l * bt * ch;
        let preatt_off = al.preatt.off + l * b * nh * t * t;
        let att_off = al.att.off + l * b * nh * t * t;
        let attproj_off = al.attproj.off + l * bt * ch;
        let residual2_off = al.residual2.off + l * bt * ch;
        let ln2_off = al.ln2.off + l * bt * ch;
        let ln2_mean_off = al.ln2_mean.off + l * bt;
        let ln2_rstd_off = al.ln2_rstd.off + l * bt;
        let fch_off = al.fch.off + l * bt * 4 * ch;
        let fch_gelu_off = al.fch_gelu.off + l * bt * 4 * ch;
        let fcproj_off = al.fcproj.off + l * bt * ch;
        let residual3_off = al.residual3.off + l * bt * ch;

        // Per-layer parameter slices.
        let ln1w = &params[pl.ln1w.off + l * ch..pl.ln1w.off + (l + 1) * ch];
        let ln1b = &params[pl.ln1b.off + l * ch..pl.ln1b.off + (l + 1) * ch];
        let qkvw = &params[pl.qkvw.off + l * 3 * ch * ch..pl.qkvw.off + (l + 1) * 3 * ch * ch];
        let qkvb = &params[pl.qkvb.off + l * 3 * ch..pl.qkvb.off + (l + 1) * 3 * ch];
        let attprojw = &params[pl.attprojw.off + l * ch * ch..pl.attprojw.off + (l + 1) * ch * ch];
        let attprojb = &params[pl.attprojb.off + l * ch..pl.attprojb.off + (l + 1) * ch];
        let ln2w = &params[pl.ln2w.off + l * ch..pl.ln2w.off + (l + 1) * ch];
        let ln2b = &params[pl.ln2b.off + l * ch..pl.ln2b.off + (l + 1) * ch];
        let fcw = &params[pl.fcw.off + l * 4 * ch * ch..pl.fcw.off + (l + 1) * 4 * ch * ch];
        let fcb = &params[pl.fcb.off + l * 4 * ch..pl.fcb.off + (l + 1) * 4 * ch];
        let fcprojw =
            &params[pl.fcprojw.off + l * ch * 4 * ch..pl.fcprojw.off + (l + 1) * ch * 4 * ch];
        let fcprojb = &params[pl.fcprojb.off + l * ch..pl.fcprojb.off + (l + 1) * ch];

        layernorm_forward(acts, ln1_off, ln1_mean_off, ln1_rstd_off, res_off, ln1w, ln1b, bt, ch);
        linear(acts, qkv_off, ln1_off, qkvw, Some(qkvb), bt, ch, 3 * ch);
        attention_forward(acts, atty_off, preatt_off, att_off, qkv_off, b, t, ch, nh);
        linear(acts, attproj_off, atty_off, attprojw, Some(attprojb), bt, ch, ch);
        residual_forward(acts, residual2_off, res_off, attproj_off, bt * ch);
        layernorm_forward(acts, ln2_off, ln2_mean_off, ln2_rstd_off, residual2_off, ln2w, ln2b, bt, ch);
        linear(acts, fch_off, ln2_off, fcw, Some(fcb), bt, ch, 4 * ch);
        gelu_forward(acts, fch_gelu_off, fch_off, bt * 4 * ch);
        linear(acts, fcproj_off, fch_gelu_off, fcprojw, Some(fcprojb), bt, 4 * ch, ch);
        residual_forward(acts, residual3_off, residual2_off, fcproj_off, bt * ch);
    }

    let resf_off = al.residual3.off + (nl - 1) * bt * ch;
    layernorm_forward(
        acts, al.lnf.off, al.lnf_mean.off, al.lnf_rstd.off, resf_off,
        &params[pl.lnfw.range()], &params[pl.lnfb.range()], bt, ch,
    );
    // lm_head is weight-tied to wte: logits = lnf @ wte^T, no bias.
    linear(acts, al.logits.off, al.lnf.off, &params[pl.wte.range()], None, bt, ch, v);

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

const GELU_SCALE: f64 = 0.797_884_560_802_865_4; // sqrt(2/pi)

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
    wpe: &[F],
    b: usize,
    t: usize,
    c: usize,
) {
    for bi in 0..b {
        for ti in 0..t {
            let ix = ids[bi * t + ti] as usize;
            let o = out_off + (bi * t + ti) * c;
            for ci in 0..c {
                acts[o + ci] = wte[ix * c + ci] + wpe[ti * c + ci];
            }
        }
    }
}

fn layernorm_forward<F: Float>(
    acts: &mut [F],
    out_off: usize,
    mean_off: usize,
    rstd_off: usize,
    inp_off: usize,
    weight: &[F],
    bias: &[F],
    n: usize,
    c: usize,
) {
    let eps = cast::<F>(1e-5);
    let cf = cast::<F>(c as f64);
    for r in 0..n {
        let base = inp_off + r * c;
        let mut m = F::zero();
        for i in 0..c {
            m = m + acts[base + i];
        }
        m = m / cf;
        let mut var = F::zero();
        for i in 0..c {
            let d = acts[base + i] - m;
            var = var + d * d;
        }
        var = var / cf;
        let rstd = F::one() / (var + eps).sqrt();
        let ob = out_off + r * c;
        for i in 0..c {
            let norm = (acts[base + i] - m) * rstd;
            acts[ob + i] = norm * weight[i] + bias[i];
        }
        acts[mean_off + r] = m;
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

/// Causal multi-head self-attention. `inp` is the packed qkv `[B, T, 3C]`.
fn attention_forward<F: Float>(
    acts: &mut [F],
    out_off: usize,
    preatt_off: usize,
    att_off: usize,
    inp_off: usize,
    b: usize,
    t: usize,
    c: usize,
    nh: usize,
) {
    let hs = c / nh;
    let scale = F::one() / cast::<F>(hs as f64).sqrt();
    let c3 = 3 * c;
    for bi in 0..b {
        for h in 0..nh {
            for ti in 0..t {
                let q_base = inp_off + (bi * t + ti) * c3 + h * hs;
                let pre_base = preatt_off + ((bi * nh + h) * t + ti) * t;
                let att_base = att_off + ((bi * nh + h) * t + ti) * t;

                // scores against keys at positions <= ti, tracking the max
                let mut maxval = F::neg_infinity();
                for t2 in 0..=ti {
                    let k_base = inp_off + (bi * t + t2) * c3 + c + h * hs;
                    let mut val = F::zero();
                    for i in 0..hs {
                        val = val + acts[q_base + i] * acts[k_base + i];
                    }
                    val = val * scale;
                    acts[pre_base + t2] = val;
                    if val > maxval {
                        maxval = val;
                    }
                }
                // softmax over the causal window
                let mut sum = F::zero();
                for t2 in 0..=ti {
                    let e = (acts[pre_base + t2] - maxval).exp();
                    acts[att_base + t2] = e;
                    sum = sum + e;
                }
                let inv = if sum > F::zero() { F::one() / sum } else { F::zero() };
                for t2 in 0..t {
                    if t2 <= ti {
                        acts[att_base + t2] = acts[att_base + t2] * inv;
                    } else {
                        // upper triangle is masked out
                        acts[att_base + t2] = F::zero();
                        acts[pre_base + t2] = F::zero();
                    }
                }
                // weighted sum of values
                let o_base = out_off + (bi * t + ti) * c + h * hs;
                for i in 0..hs {
                    acts[o_base + i] = F::zero();
                }
                for t2 in 0..=ti {
                    let v_base = inp_off + (bi * t + t2) * c3 + 2 * c + h * hs;
                    let a = acts[att_base + t2];
                    for i in 0..hs {
                        acts[o_base + i] = acts[o_base + i] + a * acts[v_base + i];
                    }
                }
            }
        }
    }
}

fn residual_forward<F: Float>(acts: &mut [F], out_off: usize, in1_off: usize, in2_off: usize, n: usize) {
    for i in 0..n {
        acts[out_off + i] = acts[in1_off + i] + acts[in2_off + i];
    }
}

/// Row-wise softmax of `logits [n, v]` into `probs [n, v]` (numerically stable).
fn softmax_forward<F: Float>(acts: &mut [F], probs_off: usize, logits_off: usize, n: usize, v: usize) {
    for r in 0..n {
        let lb = logits_off + r * v;
        let pb = probs_off + r * v;
        let mut maxval = F::neg_infinity();
        for i in 0..v {
            if acts[lb + i] > maxval {
                maxval = acts[lb + i];
            }
        }
        let mut sum = F::zero();
        for i in 0..v {
            let e = (acts[lb + i] - maxval).exp();
            acts[pb + i] = e;
            sum = sum + e;
        }
        let inv = F::one() / sum;
        for i in 0..v {
            acts[pb + i] = acts[pb + i] * inv;
        }
    }
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

/// Tanh-approximation GELU (the GPT-2 variant).
fn gelu_forward<F: Float>(acts: &mut [F], out_off: usize, inp_off: usize, n: usize) {
    for i in 0..n {
        let x = acts[inp_off + i];
        let cube = cast::<F>(0.044715) * x * x * x;
        acts[out_off + i] = cast::<F>(0.5) * x * (F::one() + (cast::<F>(GELU_SCALE) * (x + cube)).tanh());
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
fn crossentropy_softmax_backward<F: Float>(
    gacts: &mut [F],
    dlogits_off: usize,
    acts: &[F],
    probs_off: usize,
    targets: &[u16],
    n: usize,
    v: usize,
    dmean: F,
) {
    for r in 0..n {
        let ix = targets[r] as usize;
        for j in 0..v {
            let ind = if j == ix { F::one() } else { F::zero() };
            let g = (acts[probs_off + r * v + j] - ind) * dmean;
            gacts[dlogits_off + r * v + j] = gacts[dlogits_off + r * v + j] + g;
        }
    }
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
fn layernorm_backward<F: Float>(
    gacts: &mut [F],
    dinp_off: usize,
    dout_off: usize,
    grads: &mut [F],
    dweight_off: usize,
    dbias_off: usize,
    acts: &[F],
    inp_off: usize,
    mean_off: usize,
    rstd_off: usize,
    params: &[F],
    weight_off: usize,
    n: usize,
    c: usize,
) {
    let cf = cast::<F>(c as f64);
    for r in 0..n {
        let mean = acts[mean_off + r];
        let rstd = acts[rstd_off + r];
        let ib = inp_off + r * c;
        let ob = dout_off + r * c;
        let dib = dinp_off + r * c;

        // two reduction terms over the row
        let mut dnorm_mean = F::zero();
        let mut dnorm_norm_mean = F::zero();
        for i in 0..c {
            let norm = (acts[ib + i] - mean) * rstd;
            let dnorm_i = params[weight_off + i] * gacts[ob + i];
            dnorm_mean = dnorm_mean + dnorm_i;
            dnorm_norm_mean = dnorm_norm_mean + dnorm_i * norm;
        }
        dnorm_mean = dnorm_mean / cf;
        dnorm_norm_mean = dnorm_norm_mean / cf;

        for i in 0..c {
            let norm = (acts[ib + i] - mean) * rstd;
            let dnorm_i = params[weight_off + i] * gacts[ob + i];
            grads[dbias_off + i] = grads[dbias_off + i] + gacts[ob + i];
            grads[dweight_off + i] = grads[dweight_off + i] + norm * gacts[ob + i];
            let dval = (dnorm_i - dnorm_mean - norm * dnorm_norm_mean) * rstd;
            gacts[dib + i] = gacts[dib + i] + dval;
        }
    }
}

/// Backward of tanh-GELU. Accumulates `dinp`.
fn gelu_backward<F: Float>(gacts: &mut [F], dinp_off: usize, dout_off: usize, acts: &[F], inp_off: usize, n: usize) {
    let s = cast::<F>(GELU_SCALE);
    let half = cast::<F>(0.5);
    let a = cast::<F>(0.044715);
    let three_a = cast::<F>(3.0 * 0.044715);
    for i in 0..n {
        let x = acts[inp_off + i];
        let cube = a * x * x * x;
        let arg = s * (x + cube);
        let tanh_out = arg.tanh();
        let cosh = arg.cosh();
        let sech2 = F::one() / (cosh * cosh);
        let local =
            half * (F::one() + tanh_out) + x * half * sech2 * s * (F::one() + three_a * x * x);
        gacts[dinp_off + i] = gacts[dinp_off + i] + local * gacts[dout_off + i];
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
fn attention_backward<F: Float>(
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
) {
    let hs = c / nh;
    let scale = F::one() / cast::<F>(hs as f64).sqrt();
    let c3 = 3 * c;
    for bi in 0..b {
        for h in 0..nh {
            for ti in 0..t {
                let att_base = att_off + ((bi * nh + h) * t + ti) * t;
                let datt_base = datt_off + ((bi * nh + h) * t + ti) * t;
                let dpreatt_base = dpreatt_off + ((bi * nh + h) * t + ti) * t;
                let dout_base = dout_off + (bi * t + ti) * c + h * hs;

                // backward through the value accumulation -> datt and dvalue
                for t2 in 0..=ti {
                    let v_base = inp_off + (bi * t + t2) * c3 + 2 * c + h * hs;
                    let dv_base = dinp_off + (bi * t + t2) * c3 + 2 * c + h * hs;
                    for i in 0..hs {
                        gacts[datt_base + t2] =
                            gacts[datt_base + t2] + acts[v_base + i] * gacts[dout_base + i];
                        gacts[dv_base + i] =
                            gacts[dv_base + i] + acts[att_base + t2] * gacts[dout_base + i];
                    }
                }
                // backward through the softmax: dpreatt = (diag(att) - att att^T) datt
                for t2 in 0..=ti {
                    for t3 in 0..=ti {
                        let ind = if t2 == t3 { F::one() } else { F::zero() };
                        let local = acts[att_base + t2] * (ind - acts[att_base + t3]);
                        gacts[dpreatt_base + t3] =
                            gacts[dpreatt_base + t3] + local * gacts[datt_base + t2];
                    }
                }
                // backward through the scaled dot product -> dquery, dkey
                for t2 in 0..=ti {
                    let q_base = inp_off + (bi * t + ti) * c3 + h * hs;
                    let dq_base = dinp_off + (bi * t + ti) * c3 + h * hs;
                    let k_base = inp_off + (bi * t + t2) * c3 + c + h * hs;
                    let dk_base = dinp_off + (bi * t + t2) * c3 + c + h * hs;
                    let dpre = gacts[dpreatt_base + t2] * scale;
                    for i in 0..hs {
                        gacts[dq_base + i] = gacts[dq_base + i] + acts[k_base + i] * dpre;
                        gacts[dk_base + i] = gacts[dk_base + i] + acts[q_base + i] * dpre;
                    }
                }
            }
        }
    }
}

/// Backward of the encoder: scatter-add into `dwte` (by token id) and `dwpe`.
fn encoder_backward<F: Float>(
    grads: &mut [F],
    dwte_off: usize,
    dwpe_off: usize,
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
                grads[dwpe_off + ti * c + ci] = grads[dwpe_off + ti * c + ci] + gacts[o + ci];
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
}
