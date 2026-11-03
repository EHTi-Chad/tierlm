import unicodedata
def ranges(pred):
    out, start = [], None
    for cp in range(0x110000):
        ok = pred(cp)
        if ok and start is None: start = cp
        if not ok and start is not None: out.append((start, cp - 1)); start = None
    if start is not None: out.append((start, 0x10FFFF))
    return out
cat = lambda cp: unicodedata.category(chr(cp))
letters = ranges(lambda cp: cat(cp).startswith("L"))
numbers = ranges(lambda cp: cat(cp).startswith("N"))
WS = {0x09,0x0A,0x0B,0x0C,0x0D,0x20,0x85,0xA0,0x1680,0x2028,0x2029,0x202F,0x205F,0x3000} | set(range(0x2000, 0x200B))
spaces = ranges(lambda cp: cp in WS)
marks = ranges(lambda cp: cat(cp).startswith("M"))   # combining marks (Qwen3.5 splitting)
# Letter case, for OpenAI's o200k splitting (gpt-oss): words split where lower case turns to upper case
upper = ranges(lambda cp: cat(cp) in ("Lu", "Lt"))
lower = ranges(lambda cp: cat(cp) == "Ll")
# BERT's WordPiece normaliser (embedding models): punctuation is split off, control characters are dropped, and
# uncased models fold each character to lower case without accents (NFD, drop combining marks, then lower case).
punct = ranges(lambda cp: cat(cp).startswith("P") or (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126))
control = ranges(lambda cp: cat(cp).startswith("C") and cp not in (9, 10, 13))
zspace = ranges(lambda cp: cat(cp) == "Zs" or cp in (9, 10, 13))
def fold(cp):
    if 0xD800 <= cp <= 0xDFFF: return None
    c = chr(cp)
    f = "".join(ch for ch in unicodedata.normalize("NFD", c) if unicodedata.category(ch) != "Mn").lower()
    return None if f == c else f
folds = [(cp, fold(cp)) for cp in range(0x110000)]
folds = [(cp, f) for cp, f in folds if f is not None and not (0xAC00 <= cp <= 0xD7A3)]   # Hangul: computed in code
with open("src/unicode_tables.h", "w") as f:
    f.write("#pragma once\n#include <cstdint>\n\nnamespace tierlm::uni {\n\nstruct Range { uint32_t lo, hi; };\n\n")
    for name, rs in (("kLetter", letters), ("kNumber", numbers), ("kSpace", spaces), ("kMark", marks), ("kUpper", upper), ("kLower", lower),
                     ("kPunct", punct), ("kControl", control), ("kZSpace", zspace)):
        f.write("inline constexpr Range %s[] = {\n" % name)
        for i in range(0, len(rs), 6):
            f.write("    " + " ".join("{0x%X,0x%X}," % r for r in rs[i:i+6]) + "\n")
        f.write("};\n\n")
    # Folds: code point -> its lower-case, accent-free form (UTF-8, possibly empty or several characters)
    f.write("struct Fold { uint32_t cp; const char* to; };\n\ninline constexpr Fold kFold[] = {\n")
    for i in range(0, len(folds), 6):
        f.write("    " + " ".join('{0x%X,"%s"},' % (cp, "".join("\\x%02X" % b for b in t.encode("utf-8"))) for cp, t in folds[i:i+6]) + "\n")
    f.write("};\n\n")
    f.write("}\n")
