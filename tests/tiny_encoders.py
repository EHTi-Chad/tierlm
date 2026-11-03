"""Embedding models without downloading any: tierlm's WordPiece tokenizer and encoder, checked against references.

1. WordPiece: trains small BERT tokenizers (uncased and cased) with Hugging Face `tokenizers`, writes each into a
   GGUF the way the converter does (word-start pieces get a leading U+2581, "##" continuations are stored bare),
   and checks that tierlm gives exactly Hugging Face's token ids on tricky strings (accents, CJK, Korean, Turkish,
   punctuation, emoji, control characters, very long words).
2. Encoders: tiny random `bert` (learned positions, separate Q/K/V with biases, exact GELU, CLS or mean pooling) and
   `nomic-bert` (rotary positions, fused QKV, SwiGLU, mean pooling) models, compared with a NumPy forward pass
   written from the published architectures (post-norm LayerNorm blocks, attention in both directions).

usage: python tests/tiny_encoders.py build    (the folder with tierlm-tok-test and tierlm-embed)
needs: pip install tokenizers gguf numpy
"""
import os, subprocess, sys, tempfile, math
import numpy as np
import gguf

build = sys.argv[1] if len(sys.argv) > 1 else "build"
exe = lambda n: os.path.join(build, n + (".exe" if os.name == "nt" else ""))
tmp = tempfile.mkdtemp()
ok = True

# ---------------------------------------------------------------- 1. WordPiece
from tokenizers import BertWordPieceTokenizer
corpus = ["The quick brown fox jumps over the lazy dog.", "Hello, world! How's it going today?",
          "Café crème, naïve résumé, déjà vu — Ünïcödé façade.", "日本語のテキストと中文文本。", "Привет, как дела?",
          "한국어 텍스트 처리 테스트", "Numbers 1234567 and 3.14159 and 2026-10-27, 10,000,000.", "İstanbul and ISTANBUL, straße",
          "emoji 😀🎉 and symbols ©®™ ±×÷", "unbelievable unbelievably believe believer believing",
          "running runner runs ran tokenization tokenizer tokens", "def f(x): return x**2  # square"] * 50
CASES = ["Hello world", "Hello, world! How's it going?", "The QUICK Brown fox.", "Café crème, naïve résumé",
         "déjà vu", "Ünïcödé", "日本語のテキスト", "中文文本测试", "한국어 텍스트", "Привет мир", "İstanbul straße ẞ",
         "unbelievably", "tokenization!!!", "a.b,c;d:e?f", "x  \n\t y", "control\u0007char\u0000s�", "",
         "emoji 😀🎉!", "superlongwordthatdoesnotexistinthevocabularyatall" * 3, "é á́b ´`",
         "(parentheses) [brackets] {braces} <angle>", "$100 & 50% off @home #tag", "ﬁ ligature Ⅻ ½",
         "Ǆ Ǉ Ǌ titlecase", "ΑΒΓ αβγ Σσς", "UPPER lower MiXeD", "x" * 101, "can't won't isn't"]

def write_wpm(path, tok, lowercase):
    vocab = tok.get_vocab()                                    # token -> id
    inv = [None] * len(vocab)
    for t, i in vocab.items(): inv[i] = t
    special = {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"}
    types = [3 if t in special else 1 for t in inv]
    toks = [t if t in special else (t[2:] if t.startswith("##") else "▁" + t) for t in inv]
    w = gguf.GGUFWriter(path, "bert")
    w.add_block_count(1)
    w.add_tokenizer_model("bert")
    w.add_token_list(toks); w.add_token_types(types)
    w.add_uint32("tokenizer.ggml.cls_token_id", vocab["[CLS]"])
    w.add_uint32("tokenizer.ggml.seperator_token_id", vocab["[SEP]"])
    w.add_uint32("tokenizer.ggml.unknown_token_id", vocab["[UNK]"])
    w.add_tensor("dummy", np.zeros(1, dtype=np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()

for lowercase in (True, False):
    tok = BertWordPieceTokenizer(lowercase=lowercase, strip_accents=lowercase, clean_text=True, handle_chinese_chars=True)
    tok.train_from_iterator(corpus, vocab_size=600, min_frequency=1, limit_alphabet=500)
    path = os.path.join(tmp, f"wpm_{'uncased' if lowercase else 'cased'}.gguf")
    write_wpm(path, tok, lowercase)
    cases = os.path.join(tmp, "cases.txt")
    with open(cases, "w", encoding="utf-8") as f:
        for c in CASES:
            ids = tok.encode(c, add_special_tokens=False).ids
            f.write(c.encode("utf-8").hex() + "\t" + " ".join(map(str, ids)) + "\n")
    r = subprocess.run([exe("tierlm-tok-test"), path, cases], capture_output=True, text=True, encoding="utf-8", errors="replace")
    label = "WordPiece " + ("uncased" if lowercase else "cased  ")
    last = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr.strip()
    print(f"{label}: {last}")
    if r.returncode != 0:
        ok = False
        print(r.stdout[:3000], r.stderr[:1000])

# ---------------------------------------------------------------- 2. Encoders
def layer_norm(x, g, b, eps):
    m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * g + b

def gelu_erf(x):
    return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))

def rope_neox(x, pos, base):                                   # x: [n, H, hd], rotate halves
    hd = x.shape[-1]; half = hd // 2
    inv = base ** (-np.arange(half, dtype=np.float64) * 2.0 / hd)
    ang = pos[:, None] * inv[None, :]
    cos, sin = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    a, b = x[..., :half], x[..., half:]
    return np.concatenate([a * cos - b * sin, a * sin + b * cos], -1)

def reference(W, cfg, ids):
    E, H, eps = cfg["E"], cfg["H"], cfg["eps"]; hd = E // H; n = len(ids)
    x = W["token_embd.weight"][ids] + W["token_types.weight"][0]
    if "position_embd.weight" in W: x = x + W["position_embd.weight"][:n]
    x = layer_norm(x, W["token_embd_norm.weight"], W["token_embd_norm.bias"], eps)
    pos = np.arange(n, dtype=np.float64)
    for l in range(cfg["L"]):
        p = f"blk.{l}."
        if p + "attn_qkv.weight" in W:
            qkv = x @ W[p + "attn_qkv.weight"].T
            q, k, v = qkv[:, :E], qkv[:, E:2 * E], qkv[:, 2 * E:]
        else:
            q = x @ W[p + "attn_q.weight"].T + W[p + "attn_q.bias"]
            k = x @ W[p + "attn_k.weight"].T + W[p + "attn_k.bias"]
            v = x @ W[p + "attn_v.weight"].T + W[p + "attn_v.bias"]
        q, k, v = q.reshape(n, H, hd), k.reshape(n, H, hd), v.reshape(n, H, hd)
        if cfg["rope"]: q, k = rope_neox(q, pos, cfg["base"]), rope_neox(k, pos, cfg["base"])
        s = np.einsum("qhd,khd->hqk", q, k) / math.sqrt(hd)
        s = np.exp(s - s.max(-1, keepdims=True)); s /= s.sum(-1, keepdims=True)
        o = np.einsum("hqk,khd->qhd", s, v).reshape(n, E) @ W[p + "attn_output.weight"].T
        if p + "attn_output.bias" in W: o = o + W[p + "attn_output.bias"]
        x = layer_norm(x + o, W[p + "attn_output_norm.weight"], W[p + "attn_output_norm.bias"], eps)
        if p + "ffn_gate.weight" in W:
            g = x @ W[p + "ffn_gate.weight"].T
            h = (g / (1 + np.exp(-g))) * (x @ W[p + "ffn_up.weight"].T)
            d = h @ W[p + "ffn_down.weight"].T
        else:
            h = gelu_erf(x @ W[p + "ffn_up.weight"].T + W[p + "ffn_up.bias"])
            d = h @ W[p + "ffn_down.weight"].T + W[p + "ffn_down.bias"]
        x = layer_norm(x + d, W[p + "layer_output_norm.weight"], W[p + "layer_output_norm.bias"], eps)
    return x[0] if cfg["pool"] == 2 else x.mean(0)

def make(arch, pool, seed):
    rng = np.random.default_rng(seed)
    E, H, L, F, V, C = 64, 4, 3, 96, 120, 64
    cfg = dict(E=E, H=H, L=L, eps=1e-12, rope=arch == "nomic-bert", base=1000.0, pool=pool)
    r = lambda *s, sc=0.15: (rng.standard_normal(s) * sc).astype(np.float32)
    W = {"token_embd.weight": r(V, E, sc=0.5), "token_types.weight": r(2, E), "token_embd_norm.weight": 1 + r(E),
         "token_embd_norm.bias": r(E)}
    if arch == "bert": W["position_embd.weight"] = r(C, E, sc=0.3)
    for l in range(L):
        p = f"blk.{l}."
        if arch == "nomic-bert":
            W[p + "attn_qkv.weight"] = r(3 * E, E)
            W[p + "ffn_up.weight"] = r(F, E); W[p + "ffn_gate.weight"] = r(F, E); W[p + "ffn_down.weight"] = r(E, F)
        else:
            for t in "qkv": W[p + f"attn_{t}.weight"] = r(E, E); W[p + f"attn_{t}.bias"] = r(E)
            W[p + "attn_output.bias"] = r(E)
            W[p + "ffn_up.weight"] = r(F, E); W[p + "ffn_up.bias"] = r(F); W[p + "ffn_down.weight"] = r(E, F); W[p + "ffn_down.bias"] = r(E)
        W[p + "attn_output.weight"] = r(E, E)
        W[p + "attn_output_norm.weight"] = 1 + r(E); W[p + "attn_output_norm.bias"] = r(E)
        W[p + "layer_output_norm.weight"] = 1 + r(E); W[p + "layer_output_norm.bias"] = r(E)
    path = os.path.join(tmp, f"{arch}_{pool}.gguf")
    w = gguf.GGUFWriter(path, arch)
    w.add_uint32(f"{arch}.embedding_length", E); w.add_uint32(f"{arch}.attention.head_count", H)
    w.add_uint32(f"{arch}.block_count", L); w.add_uint32(f"{arch}.context_length", C)
    w.add_uint32(f"{arch}.feed_forward_length", F)
    w.add_float32(f"{arch}.attention.layer_norm_epsilon", cfg["eps"]); w.add_bool(f"{arch}.attention.causal", False)
    w.add_uint32(f"{arch}.pooling_type", pool)
    if arch == "nomic-bert": w.add_float32(f"{arch}.rope.freq_base", cfg["base"])
    for name, t in W.items(): w.add_tensor(name, t)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    return path, W, cfg

for arch, pool in (("bert", 2), ("bert", 1), ("nomic-bert", 1)):
    path, W, cfg = make(arch, pool, 7 if arch == "bert" else 11)
    worst = 0.0
    for n in (1, 5, 37, 64):
        ids = list(np.random.default_rng(n).integers(0, 120, n))
        ref = reference({k: v.astype(np.float64) for k, v in W.items()}, cfg, np.array(ids))
        r = subprocess.run([exe("tierlm-embed"), path, "--ids", ",".join(map(str, ids)), "-d", "cpu", "-t", "2"],
                           capture_output=True, text=True)
        if r.returncode != 0: print(r.stderr); ok = False; break
        got = np.array([float(x) for x in r.stdout.split("embedding:")[1].split()])
        worst = max(worst, float(np.abs(got - ref).max() / (np.abs(ref).max() + 1e-12)))
    good = worst < 1e-4
    ok &= good
    print(f"{arch:10s} pooling {'cls ' if pool == 2 else 'mean'}: worst relative diff {worst:.1e} (limit 1e-04)  ->  {'ok' if good else 'FAIL'}")

print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
