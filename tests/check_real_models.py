"""Checks tierlm's tokenizer on real GGUF files against the official Hugging Face tokenizer of each model.

usage: python tests/check_real_models.py build/tierlm-tok-test model.gguf=hf/repo [model.gguf=hf/repo ...]
  e.g. python tests/check_real_models.py build/tierlm-tok-test \\
         ~/models/Qwen3-0.6B-Q8_0.gguf=Qwen/Qwen3-0.6B \\
         ~/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf=unsloth/Llama-3.2-1B-Instruct \\
         ~/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf=TinyLlama/TinyLlama-1.1B-Chat-v1.0
needs: pip install tokenizers huggingface_hub   (downloads only each model's tokenizer.json, a few MB)
"""
import os, subprocess, sys, tempfile
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

CASES = ["Hello world", "Hello, world! How's it going?", "I'M HERE and you're THERE; we'll see, they'd go.",
         "  leading spaces", "trailing spaces   ", "multiple   spaces   between", "tabs\tand\ttabs\t\t",
         "line1\nline2\n\nline4\r\nwindows", "\n\n  indented after blank lines", "12345678901234", "pi is 3.14159265",
         "2026-10-27 at 16:30", "x=1;y=22;z=333;w=4444", "def f(x):\n    return x**2  # square\n",
         "Ça va? Être naïve, déjà vu.", "日本語のテキストと中文文本。", "Привет мир", "emoji 😀🎉👍🏽!", "'quoted' and \"double\"",
         "a'b'c 's 't", "...!!!???", "URL: https://example.com/a?b=c&d=e", "mixed123abc456",
         "ALLCAPS and lowercase And Title", "x  \n  y", "word\n", "über straße", "١٢٣ arabic digits",
         "The quick brown fox jumps over the lazy dog. " * 3]

tool, ok = sys.argv[1], True
for arg in sys.argv[2:]:
    path, repo = arg.rsplit("=", 1)
    tk = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json"))
    cf = os.path.join(tempfile.mkdtemp(), "cases.txt")
    with open(cf, "w") as f:
        for c in CASES: f.write(c.encode().hex() + "\t" + " ".join(map(str, tk.encode(c, add_special_tokens=False).ids)) + "\n")
    r = subprocess.run([tool, os.path.expanduser(path), cf], capture_output=True, text=True)
    ok &= r.returncode == 0
    print(f"== {os.path.basename(path)} vs {repo}")
    print((r.stdout + r.stderr).strip())
print("PASS" if ok else "FAIL")
