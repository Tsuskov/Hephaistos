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

use rand::Rng;

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

/// A GPT-2-style transformer: flat params + activation arena.
pub struct Gpt {
    pub cfg: Config,
    params: Vec<f32>,
    acts: Vec<f32>,
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
        Self { cfg, params, acts, pl, al }
    }

    pub fn num_params(&self) -> usize {
        self.params.len()
    }

    /// Logits `[B, T, V]` from the last forward pass.
    pub fn logits(&self) -> &[f32] {
        &self.acts[self.al.logits.range()]
    }

    /// Run the forward pass for one batch of token ids (`len == B*T`).
    ///
    /// If `targets` is given, also computes softmax `probs` + per-token
    /// cross-entropy `losses` and returns the mean loss over `B*T`.
    pub fn forward(&mut self, ids: &[u16], targets: Option<&[u16]>) -> Option<f32> {
        let c = self.cfg;
        let (b, t, ch, nh, v, nl) =
            (c.batch_size, c.block_size, c.n_embd, c.n_head, c.vocab_size, c.n_layer);
        let bt = b * t;
        assert_eq!(ids.len(), bt, "ids must be batch_size * block_size");
        assert!(t <= c.block_size, "block_size exceeded");

        let pl = self.pl;
        let al = self.al;

        encoder_forward(
            &mut self.acts,
            al.encoded.off,
            ids,
            &self.params[pl.wte.range()],
            &self.params[pl.wpe.range()],
            b, t, ch,
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
            let ln1w = &self.params[pl.ln1w.off + l * ch..pl.ln1w.off + (l + 1) * ch];
            let ln1b = &self.params[pl.ln1b.off + l * ch..pl.ln1b.off + (l + 1) * ch];
            let qkvw =
                &self.params[pl.qkvw.off + l * 3 * ch * ch..pl.qkvw.off + (l + 1) * 3 * ch * ch];
            let qkvb = &self.params[pl.qkvb.off + l * 3 * ch..pl.qkvb.off + (l + 1) * 3 * ch];
            let attprojw =
                &self.params[pl.attprojw.off + l * ch * ch..pl.attprojw.off + (l + 1) * ch * ch];
            let attprojb =
                &self.params[pl.attprojb.off + l * ch..pl.attprojb.off + (l + 1) * ch];
            let ln2w = &self.params[pl.ln2w.off + l * ch..pl.ln2w.off + (l + 1) * ch];
            let ln2b = &self.params[pl.ln2b.off + l * ch..pl.ln2b.off + (l + 1) * ch];
            let fcw =
                &self.params[pl.fcw.off + l * 4 * ch * ch..pl.fcw.off + (l + 1) * 4 * ch * ch];
            let fcb = &self.params[pl.fcb.off + l * 4 * ch..pl.fcb.off + (l + 1) * 4 * ch];
            let fcprojw =
                &self.params[pl.fcprojw.off + l * ch * 4 * ch..pl.fcprojw.off + (l + 1) * ch * 4 * ch];
            let fcprojb =
                &self.params[pl.fcprojb.off + l * ch..pl.fcprojb.off + (l + 1) * ch];

            layernorm_forward(
                &mut self.acts, ln1_off, ln1_mean_off, ln1_rstd_off, res_off, ln1w, ln1b, bt, ch,
            );
            matmul_forward(&mut self.acts, qkv_off, ln1_off, qkvw, Some(qkvb), bt, ch, 3 * ch);
            attention_forward(
                &mut self.acts, atty_off, preatt_off, att_off, qkv_off, b, t, ch, nh,
            );
            matmul_forward(&mut self.acts, attproj_off, atty_off, attprojw, Some(attprojb), bt, ch, ch);
            residual_forward(&mut self.acts, residual2_off, res_off, attproj_off, bt * ch);
            layernorm_forward(
                &mut self.acts, ln2_off, ln2_mean_off, ln2_rstd_off, residual2_off, ln2w, ln2b, bt, ch,
            );
            matmul_forward(&mut self.acts, fch_off, ln2_off, fcw, Some(fcb), bt, ch, 4 * ch);
            gelu_forward(&mut self.acts, fch_gelu_off, fch_off, bt * 4 * ch);
            matmul_forward(
                &mut self.acts, fcproj_off, fch_gelu_off, fcprojw, Some(fcprojb), bt, 4 * ch, ch,
            );
            residual_forward(&mut self.acts, residual3_off, residual2_off, fcproj_off, bt * ch);
        }

        let resf_off = al.residual3.off + (nl - 1) * bt * ch;
        layernorm_forward(
            &mut self.acts,
            al.lnf.off, al.lnf_mean.off, al.lnf_rstd.off, resf_off,
            &self.params[pl.lnfw.range()], &self.params[pl.lnfb.range()], bt, ch,
        );
        // lm_head is weight-tied to wte: logits = lnf @ wte^T, no bias.
        matmul_forward(&mut self.acts, al.logits.off, al.lnf.off, &self.params[pl.wte.range()], None, bt, ch, v);

        // Loss (optional).
        targets.map(|targets| {
            assert_eq!(targets.len(), bt, "targets must be batch_size * block_size");
            softmax_forward(&mut self.acts, al.probs.off, al.logits.off, bt, v);
            crossentropy_forward(&mut self.acts, al.losses.off, al.probs.off, targets, bt, v);
            let losses = &self.acts[al.losses.range()];
            losses.iter().sum::<f32>() / bt as f32
        })
    }
}

/// Standard normal sample via Box–Muller.
fn randn<R: Rng>(rng: &mut R) -> f32 {
    let u1: f32 = rng.r#gen::<f32>().max(1e-7);
    let u2: f32 = rng.r#gen::<f32>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

const GELU_SCALE: f32 = 0.797_884_56; // sqrt(2/pi)

fn encoder_forward(
    acts: &mut [f32],
    out_off: usize,
    ids: &[u16],
    wte: &[f32],
    wpe: &[f32],
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

fn layernorm_forward(
    acts: &mut [f32],
    out_off: usize,
    mean_off: usize,
    rstd_off: usize,
    inp_off: usize,
    weight: &[f32],
    bias: &[f32],
    n: usize,
    c: usize,
) {
    let eps = 1e-5f32;
    for r in 0..n {
        let base = inp_off + r * c;
        let mut m = 0.0f32;
        for i in 0..c {
            m += acts[base + i];
        }
        m /= c as f32;
        let mut var = 0.0f32;
        for i in 0..c {
            let d = acts[base + i] - m;
            var += d * d;
        }
        var /= c as f32;
        let rstd = 1.0 / (var + eps).sqrt();
        let ob = out_off + r * c;
        for i in 0..c {
            let norm = (acts[base + i] - m) * rstd;
            acts[ob + i] = norm * weight[i] + bias[i];
        }
        acts[mean_off + r] = m;
        acts[rstd_off + r] = rstd;
    }
}

/// Linear layer with weight `[OC, C]` (row = output unit): `out = inp @ W^T + b`.
fn matmul_forward(
    acts: &mut [f32],
    out_off: usize,
    inp_off: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    n: usize,
    c: usize,
    oc: usize,
) {
    for r in 0..n {
        let ib = inp_off + r * c;
        let ob = out_off + r * oc;
        for o in 0..oc {
            let mut val = bias.map_or(0.0, |b| b[o]);
            let wb = o * c;
            for i in 0..c {
                val += acts[ib + i] * weight[wb + i];
            }
            acts[ob + o] = val;
        }
    }
}

/// Causal multi-head self-attention. `inp` is the packed qkv `[B, T, 3C]`.
fn attention_forward(
    acts: &mut [f32],
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
    let scale = 1.0 / (hs as f32).sqrt();
    let c3 = 3 * c;
    for bi in 0..b {
        for h in 0..nh {
            for ti in 0..t {
                let q_base = inp_off + (bi * t + ti) * c3 + h * hs;
                let pre_base = preatt_off + ((bi * nh + h) * t + ti) * t;
                let att_base = att_off + ((bi * nh + h) * t + ti) * t;

                // scores against keys at positions <= ti, tracking the max
                let mut maxval = f32::NEG_INFINITY;
                for t2 in 0..=ti {
                    let k_base = inp_off + (bi * t + t2) * c3 + c + h * hs;
                    let mut val = 0.0f32;
                    for i in 0..hs {
                        val += acts[q_base + i] * acts[k_base + i];
                    }
                    val *= scale;
                    acts[pre_base + t2] = val;
                    if val > maxval {
                        maxval = val;
                    }
                }
                // softmax over the causal window
                let mut sum = 0.0f32;
                for t2 in 0..=ti {
                    let e = (acts[pre_base + t2] - maxval).exp();
                    acts[att_base + t2] = e;
                    sum += e;
                }
                let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
                for t2 in 0..t {
                    if t2 <= ti {
                        acts[att_base + t2] *= inv;
                    } else {
                        // upper triangle is masked out
                        acts[att_base + t2] = 0.0;
                        acts[pre_base + t2] = 0.0;
                    }
                }
                // weighted sum of values
                let o_base = out_off + (bi * t + ti) * c + h * hs;
                for i in 0..hs {
                    acts[o_base + i] = 0.0;
                }
                for t2 in 0..=ti {
                    let v_base = inp_off + (bi * t + t2) * c3 + 2 * c + h * hs;
                    let a = acts[att_base + t2];
                    for i in 0..hs {
                        acts[o_base + i] += a * acts[v_base + i];
                    }
                }
            }
        }
    }
}

fn residual_forward(acts: &mut [f32], out_off: usize, in1_off: usize, in2_off: usize, n: usize) {
    for i in 0..n {
        acts[out_off + i] = acts[in1_off + i] + acts[in2_off + i];
    }
}

/// Row-wise softmax of `logits [n, v]` into `probs [n, v]` (numerically stable).
fn softmax_forward(acts: &mut [f32], probs_off: usize, logits_off: usize, n: usize, v: usize) {
    for r in 0..n {
        let lb = logits_off + r * v;
        let pb = probs_off + r * v;
        let mut maxval = f32::NEG_INFINITY;
        for i in 0..v {
            if acts[lb + i] > maxval {
                maxval = acts[lb + i];
            }
        }
        let mut sum = 0.0f32;
        for i in 0..v {
            let e = (acts[lb + i] - maxval).exp();
            acts[pb + i] = e;
            sum += e;
        }
        let inv = 1.0 / sum;
        for i in 0..v {
            acts[pb + i] *= inv;
        }
    }
}

/// Per-token cross-entropy `losses[r] = -ln(probs[r, target[r]])`.
fn crossentropy_forward(
    acts: &mut [f32],
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
fn gelu_forward(acts: &mut [f32], out_off: usize, inp_off: usize, n: usize) {
    for i in 0..n {
        let x = acts[inp_off + i];
        let cube = 0.044715 * x * x * x;
        acts[out_off + i] = 0.5 * x * (1.0 + (GELU_SCALE * (x + cube)).tanh());
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
}
