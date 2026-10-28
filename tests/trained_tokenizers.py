"""Tokenizer test without downloading any model: trains small tokenizers with Hugging Face `tokenizers`
(byte-level BPE with the GPT-2, Llama 3 and Qwen2 split patterns) and Google `sentencepiece` (Llama 2 /
Mistral style BPE with byte fallback), writes each one into a GGUF, and checks that tierlm produces exactly
the same token ids on a set of tricky strings, and decodes them back to the same text.

usage: python tests/trained_tokenizers.py build/tierlm-tok-test
needs: pip install tokenizers sentencepiece gguf numpy
"""
import os, subprocess, sys, tempfile, glob
import numpy as np
import gguf

here = os.path.dirname(os.path.abspath(__file__))
corpus = []
for f in glob.glob(os.path.join(here, "..", "docs", "*.md")) + glob.glob(os.path.join(here, "..", "src", "*.cpp")):
    corpus += open(f, encoding="utf-8").read().splitlines()
corpus += ["Ça va? Être naïve, déjà vu — café crème.", "日本語のテキストと中文文本。", "Привет, как дела?",
           "Numbers 1234567 and 3.14159 and 2026-10-27, 10,000,000.", "I'M HERE, YOU'RE THERE, WE'LL SEE, THEY'D GO.",
           "emoji 😀🎉👍🏽 and symbols ©®™ ±×÷", "\t\ttabs\tand  double  spaces   triple", "line1\nline2\r\nline3\n\n\nend"] * 20

CASES = ["Hello world", "Hello, world! How's it going?", "I'M HERE and you're THERE; we'll see, they'd go.",
         "  leading spaces", "trailing spaces   ", "multiple   spaces   between", "tabs\tand\ttabs\t\t",
         "line1\nline2\n\nline4\r\nwindows", "\n\n  indented after blank lines", "12345678901234", "pi is 3.14159265",
         "2026-10-27 at 16:30", "x=1;y=22;z=333;w=4444", "def f(x):\n    return x**2  # square\n",
         "Ça va? Être naïve, déjà vu.", "日本語のテキスト", "Привет мир", "emoji 😀🎉👍🏽!", "'quoted' and \"double\"",
         "a'b'c 's 't", "<|im_start|>user\nhi<|im_end|>", "", " ", "\n", "...!!!???", "URL: https://example.com/a?b=c&d=e",
         "mixed123abc456", "ALLCAPS and lowercase And Title", "x  \n  y", "word\n", "über straße", "١٢٣ arabic digits",
         "cafe\u0301 nai\u0308ve", "हिन्दी भाषा", "ภาษาไทย ทดสอบ", "\u0301x .\u0301 a\u0301\u0301b"]

def write_gguf(path, pre, tokens, types, merges=None, scores=None, model="gpt2", bos=None, eos=None):
    w = gguf.GGUFWriter(path, "llama")
    w.add_block_count(1)
    w.add_tokenizer_model(model)
    if pre is not None: w.add_tokenizer_pre(pre)
    w.add_token_list(tokens); w.add_token_types(types)
    if merges is not None: w.add_token_merges(merges)
    if scores is not None: w.add_token_scores(scores)
    if bos is not None: w.add_bos_token_id(bos)
    if eos is not None: w.add_eos_token_id(eos)
    w.add_tensor("dummy", np.zeros(1, dtype=np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()

def run(tool, gguf_path, cases, ids):
    cf = gguf_path + ".cases.txt"
    with open(cf, "w", encoding="utf-8") as f:
        for t, i in zip(cases, ids): f.write(t.encode().hex() + "\t" + " ".join(map(str, i)) + "\n")
    r = subprocess.run([tool, gguf_path, cf], capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.returncode == 0, (r.stdout + r.stderr).strip()

tool = sys.argv[1]
tmp = tempfile.mkdtemp()
ok = True

# ---- byte-level BPE with three split patterns ----
from tokenizers import Tokenizer, Regex, models, pre_tokenizers, trainers, decoders, AddedToken
PATTERNS = {
    "gpt-2":     r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
    "llama-bpe": r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    "qwen2":     r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    "qwen35":    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
}
for pre, pat in PATTERNS.items():
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Split(Regex(pat), behavior="isolated", invert=False),
                                                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tk.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=1500, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False,
                             special_tokens=[AddedToken("<|im_start|>", special=True), AddedToken("<|im_end|>", special=True)])
    tk.train_from_iterator(corpus, tr)
    vocab = tk.get_vocab()
    tokens = [None] * len(vocab)
    for t, i in vocab.items(): tokens[i] = t
    types = [3 if t in ("<|im_start|>", "<|im_end|>") else 1 for t in tokens]
    import json
    merges = [" ".join(m) if isinstance(m, list) else m for m in json.loads(tk.to_str())["model"]["merges"]]
    p = os.path.join(tmp, f"bpe-{pre}.gguf")
    write_gguf(p, pre, tokens, types, merges=merges)
    # tierlm's encode() leaves control tokens as text unless asked; Hugging Face splits them out always,
    # so compare on text without control-token strings for the plain cases.
    cases = [c for c in CASES if "<|" not in c]
    ids = [tk.encode(c).ids for c in cases]
    good, out = run(tool, p, cases, ids)
    ok &= good
    print(f"BPE {pre:10s}: {out.splitlines()[-1] if out else ''}" + ("" if good else "\n" + out))

# ---- SentencePiece BPE (Llama 2 / Mistral 7B style) ----
import sentencepiece as spm
cf = os.path.join(tmp, "corpus.txt")
open(cf, "w", encoding="utf-8").write("\n".join(corpus))
prefix = os.path.join(tmp, "sp")
spm.SentencePieceTrainer.train(input=cf, model_prefix=prefix, vocab_size=1200, model_type="bpe", byte_fallback=True,
                               normalization_rule_name="identity", remove_extra_whitespaces=False, add_dummy_prefix=True,
                               split_digits=True, allow_whitespace_only_pieces=True, character_coverage=0.9995,
                               user_defined_symbols=[], minloglevel=2)
sp = spm.SentencePieceProcessor(model_file=prefix + ".model")
tokens, scores, types = [], [], []
for i in range(sp.get_piece_size()):
    tokens.append(sp.id_to_piece(i)); scores.append(sp.get_score(i))
    types.append(2 if sp.is_unknown(i) else 3 if sp.is_control(i) else 6 if sp.is_byte(i) else 1)
p = os.path.join(tmp, "spm.gguf")
write_gguf(p, None, tokens, types, scores=scores, model="llama", bos=sp.bos_id(), eos=sp.eos_id())
cases = [c for c in CASES if "<|" not in c and c.strip() != "" ]   # SentencePiece drops a bare-space input
ids = [sp.encode(c) for c in cases]
good, out = run(tool, p, cases, ids)
ok &= good
print(f"SentencePiece : {out.splitlines()[-1] if out else ''}" + ("" if good else "\n" + out))

# Same SentencePiece model stored the way Hugging Face conversions often do it: every score 0, and the merge
# order in a merges list instead (built like Hugging Face's SentencePieceExtractor).
vocab = {sp.id_to_piece(i): i for i in range(sp.get_piece_size())}
merges = []
for piece, pid in vocab.items():
    if types[pid] != 1: continue
    for k in range(1, len(piece)):
        a, b = piece[:k], piece[k:]
        if a in vocab and b in vocab: merges.append((pid, vocab[a], vocab[b], a + " " + b))
merges.sort()
p = os.path.join(tmp, "spm-merges.gguf")
write_gguf(p, None, tokens, types, scores=[0.0] * len(tokens), merges=[m[3] for m in merges], model="llama",
           bos=sp.bos_id(), eos=sp.eos_id())
good, out = run(tool, p, cases, ids)
ok &= good
print(f"SentencePiece, merges list: {out.splitlines()[-1] if out else ''}" + ("" if good else "\n" + out))
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
