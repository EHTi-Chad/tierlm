#!/usr/bin/env python3
"""Builds web/bench-data.js: the fixed questions of the benchmark page (/bench).

Every run asks the same questions, so scores compare across models and PCs. The sources (all free to share):
  GSM8K test set, MIT (openai/grade-school-math)          -> 10 maths word problems
  MMLU test set, MIT (cais/mmlu, all/test parquet)          -> 5 multiple choice
  ARC-Challenge test set, CC BY-SA 4.0 (allenai/ai2_arc)    -> 5 multiple choice
  HumanEval, MIT (openai/human-eval)                        -> 5 functions, asked for in JavaScript, tests translated
  IFEval prompts, Apache-2.0 (google-research)              -> 5 prompts with one checkable instruction each
Usage: make-bench-data.py gsm8k-test.jsonl HumanEval.jsonl ifeval-input_data.jsonl mmlu-test.parquet arc-challenge-test.parquet
"""
import json, sys

gsm_path, he_path, if_path, mmlu_path, arc_path = sys.argv[1:6]

# GSM8K: the first 10 problems of the test set whose question is under 250 characters (short to read, same every time)
gsm = []
for line in open(gsm_path, encoding="utf-8"):
    r = json.loads(line)
    if len(r["question"]) < 250:
        gsm.append({"q": r["question"], "a": r["answer"].split("####")[-1].strip().replace(",", "")})
    if len(gsm) == 10: break

# HumanEval: 5 problems, from easy to harder; the model writes them in JavaScript (tests translated by hand below)
he = {json.loads(l)["task_id"]: json.loads(l) for l in open(he_path, encoding="utf-8")}
tests = {
    "HumanEval/0": ["f([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) === true", "f([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) === false",
                    "f([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) === true", "f([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) === false",
                    "f([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) === true", "f([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) === true",
                    "f([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) === false"],
    "HumanEval/12": ["f([]) === null || f([]) === undefined", "f(['x', 'y', 'z']) === 'x'",
                     "f(['x', 'yyy', 'zzzz', 'www', 'kkkk', 'abc']) === 'zzzz'"],
    "HumanEval/26": ["same(f([]), [])", "same(f([1, 2, 3, 4]), [1, 2, 3, 4])", "same(f([1, 2, 3, 2, 4, 3, 5]), [1, 4, 5])"],
    "HumanEval/46": ["f(5) === 4", "f(8) === 28", "f(10) === 104", "f(12) === 386"],
    "HumanEval/107": ["same(f(123), [8, 13])", "same(f(12), [4, 6])", "same(f(3), [1, 2])", "same(f(63), [6, 8])",
                      "same(f(25), [5, 6])", "same(f(19), [4, 6])", "same(f(9), [4, 5])", "same(f(1), [0, 1])"],
}
code = [{"id": k, "name": he[k]["entry_point"], "spec": he[k]["prompt"].strip(), "tests": t} for k, t in tests.items()]

# IFEval: one prompt for each of five instructions a page can check exactly
keys = {3615: "no_comma", 202: "lowercase", 2314: "bullets", 3001: "end_phrase", 3223: "json"}
ife = []
for line in open(if_path, encoding="utf-8"):
    r = json.loads(line)
    if r["key"] in keys:
        ife.append({"key": r["key"], "check": keys[r["key"]], "prompt": r["prompt"], "kwargs": r["kwargs"][0]})
ife.sort(key=lambda r: list(keys).index(r["key"]))

# MMLU and ARC-Challenge: 5 each, spread through the set (every n-th row), so not all from one subject
import pyarrow.parquet as pq
mm = pq.read_table(mmlu_path).to_pylist()
step = len(mm) // 5
mmlu = [{"q": r["question"], "choices": list(r["choices"]), "a": "ABCD"[r["answer"]], "subject": r["subject"]}
        for r in (mm[i * step + step // 2] for i in range(5))]
ar = pq.read_table(arc_path).to_pylist()
step = len(ar) // 5
arc = []
for i in range(5):
    r = ar[i * step + step // 2]
    labels = list(r["choices"]["label"])
    arc.append({"q": r["question"], "choices": list(r["choices"]["text"]), "a": "ABCDE"[labels.index(r["answerKey"])]})

data = {"version": 1, "gsm8k": gsm, "mmlu": mmlu, "arc": arc, "code": code, "ifeval": ife}
with open("web/bench-data.js", "w", encoding="utf-8") as f:
    f.write("// The benchmark page's fixed questions, made by scripts/make-bench-data.py. Sources and licences:\n"
            "// GSM8K (MIT, OpenAI), MMLU (MIT, Hendrycks et al.), ARC-Challenge (CC BY-SA 4.0, Allen Institute for AI),\n"
            "// HumanEval (MIT, OpenAI; tests translated to JavaScript), IFEval prompts (Apache-2.0, Google).\n"
            "window.BENCH_DATA = " + json.dumps(data, ensure_ascii=False, indent=1) + ";\n")
print("gsm8k", len(gsm), "mmlu", len(mmlu), "arc", len(arc), "code", len(code), "ifeval", len(ife))
