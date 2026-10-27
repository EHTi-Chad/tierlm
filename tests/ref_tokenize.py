import json
from tokenizers import Tokenizer

tok = Tokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924-Instruct")
cases = [
    "Hello world",
    " leading space",
    "multiple   spaces   here",
    "It's they're we've I'm you'll he'd",
    "Numbers 12345 and 3.14159",
    "line one\nline two\n\nline four",
    "tabs\tand\ttrailing   ",
    "Café naïve résumé",
    "日本語のテキスト",
    "emoji 🚀🔥 test",
    "def f(x):\n    return x**2  # square",
    "",
    "   ",
]
with open("tests/tok_cases.jsonl", "w", encoding="utf-8") as f:
    for c in cases:
        f.write(json.dumps({"text": c, "ids": tok.encode(c, add_special_tokens=False).ids}, ensure_ascii=False) + "\n")
print(f"wrote {len(cases)} cases")
print(open("tests/tok_cases.jsonl", encoding="utf-8").readline().strip())
