//! Verifies the Cadmus drop-in against the *real* artifacts on disk: it must
//! load the HF `tokenizer.json` files that the old `tokenizers` crate wrote, and
//! decode the existing HF-encoded `.bin` streams back into readable text.

use std::fs;

use cadmus::BpeModel;

fn read_u16_le(path: &str) -> Vec<u16> {
    let bytes = fs::read(path).unwrap();
    bytes.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}

#[test]
fn loads_hf_tokenizers_and_decodes_existing_bins() {
    for (tok_path, bin_path) in
        [("data/tokenizer.json", "data/train.bin"), ("data/greek_tokenizer.json", "data/greek_train.bin")]
    {
        let json = fs::read_to_string(tok_path).unwrap();
        let tok = BpeModel::from_hf_json(&json).expect("cadmus loads HF tokenizer.json");
        assert!(tok.vocab_size() > 256, "{tok_path}: vocab should have learned merges");

        // Decode the first 200 tokens of the existing (HF-encoded) stream.
        let ids: Vec<u32> = read_u16_le(bin_path).iter().take(200).map(|&x| x as u32).collect();
        let text = tok.decode(&ids);
        assert!(!text.is_empty(), "{bin_path}: decoded text is empty");
        // Byte-level BPE decodes to valid UTF-8; a corrupt load would produce
        // mostly replacement chars.
        let bad = text.chars().filter(|&c| c == '\u{FFFD}').count();
        assert!(bad * 10 < text.chars().count(), "{bin_path}: too many replacement chars");
    }
}
