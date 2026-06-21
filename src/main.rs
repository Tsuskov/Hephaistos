//! Hephaistos — a GPT/Llama implemented from scratch in Rust.
//!
//! Phase 0: project scaffold. A `Config` describing the model, and a `matmul`
//! triple-loop that everything else will be built on.

/// Model hyperparameters. Weights live as flat `Vec<f32>` buffers shaped by these.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)] // fields wired up in later phases
pub struct Config {
    pub n_layer: usize,
    pub n_head: usize,
    pub n_embd: usize,
    pub block_size: usize,
    pub vocab_size: usize,
    pub batch_size: usize,
}

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

fn main() {
    // A tiny sanity demo so `cargo run` shows something real.
    let a = [1.0, 2.0, 3.0, 4.0]; // 2x2
    let b = [5.0, 6.0, 7.0, 8.0]; // 2x2
    let mut out = [0.0f32; 4];
    matmul(&mut out, &a, &b, 2, 2, 2);
    println!("matmul([[1,2],[3,4]] x [[5,6],[7,8]]) = {out:?}");
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
