"""Gemma 4's chat format and tool calls against its own chat template.

Renders conversations (thinking on and off, a system message, earlier replies with their thinking, tools with many
kinds of parameters, tool calls and their results, several calls at once, the answer after the results) with Gemma 4's
chat template run through Jinja, and checks that tierlm builds the same prompt, character for character. Then reads
tool calls back out of replies written in Gemma 4's syntax.

The template comes from a Gemma 4 GGUF (or a .jinja file): the 26B and 31B ones start the reply after an empty thought
channel when thinking is off, the E2B and E4B ones don't. Several can be given.

usage: python tests/gemma4_chat_test.py build/tierlm-chat-check gemma-4-E4B-it-Q4_K_M.gguf [gemma-4-26B-A4B-it.gguf ...]
"""
import json, subprocess, sys
import jinja2

def load_template(path):
    if path.endswith(".jinja"): return open(path, encoding="utf-8").read()
    import gguf
    return gguf.GGUFReader(path).fields["tokenizer.chat_template"].contents()

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string", "description": "The city, e.g. Toronto"},
        "unit": {"type": "string", "enum": ["c", "f"], "description": "Temperature unit"},
        "days": {"type": "integer", "description": "How many days of forecast"}},
        "required": ["city"]}}}
SEARCH = {"type": "function", "function": {
    "name": "search_files", "description": "Search the project's files.",
    "parameters": {"type": "object", "properties": {
        "query": {"type": "string"},
        "paths": {"type": "array", "items": {"type": "string"}, "description": "Folders to look in"},
        "regex": {"type": "boolean", "nullable": True},
        "limit": {"type": "number"},
        "options": {"type": "object", "description": "Extra settings",
                    "properties": {"case": {"type": "boolean"}, "depth": {"type": "integer"}}, "required": ["case"]},
        "filters": {"type": "array", "items": {"type": "object", "properties": {"ext": {"type": "string"}, "min_size": {"type": "integer"}},
                                               "required": ["ext"]}}},
        "required": ["query", "paths"]}}}
NOARGS = {"type": "function", "function": {"name": "get_time", "description": "The current time.",
                                           "parameters": {"type": "object", "properties": {}}}}
TOOLS = [WEATHER, SEARCH, NOARGS]

def call(id_, name, args): return {"id": id_, "type": "function", "function": {"name": name, "arguments": args}}

U = {"role": "user", "content": "What's the weather in Toronto?"}
CASES = [
    dict(messages=[{"role": "user", "content": "Hello!"}]),
    dict(messages=[{"role": "system", "content": "  Be brief.  "}, {"role": "user", "content": " Hi "},
                   {"role": "assistant", "content": "<|channel>thought\nThe user greets.<channel|>Hello! How can I help?"},
                   {"role": "user", "content": "Tell me a joke."}],
         # (tierlm keeps an earlier reply's thinking as <think>...</think>)
         tierlm=[{"role": "system", "content": "  Be brief.  "}, {"role": "user", "content": " Hi "},
                 {"role": "assistant", "content": "<think>The user greets.</think>\n\nHello! How can I help?"},
                 {"role": "user", "content": "Tell me a joke."}]),
    # A tool-call chain with thinking: the reasoning goes back in the thought channel (tierlm: <think> in the text,
    # as the server puts reasoning_content).
    dict(messages=[U, {"role": "assistant", "reasoning_content": "I should look it up.", "content": "",
                       "tool_calls": [call("c1", "get_weather", {"city": "Toronto"})]},
                   {"role": "tool", "tool_call_id": "c1", "content": "3C"},
                   {"role": "assistant", "reasoning_content": "Got it.", "content": "It's 3C."}], tools=TOOLS,
         tierlm=[U, {"role": "assistant", "content": "<think>\nI should look it up.\n</think>\n\n",
                     "tool_calls": [call("c1", "get_weather", {"city": "Toronto"})]},
                 {"role": "tool", "tool_call_id": "c1", "content": "3C"},
                 {"role": "assistant", "content": "<think>\nGot it.\n</think>\n\nIt's 3C."}]),
    dict(messages=[U], tools=TOOLS),
    dict(messages=[{"role": "system", "content": "You are a coding agent."}, U], tools=TOOLS),
    dict(messages=[U, {"role": "assistant", "content": "", "tool_calls": [call("c1", "get_weather", {"city": "Toronto", "unit": "c"})]},
                   {"role": "tool", "tool_call_id": "c1", "content": "{\"temp\": 3, \"sky\": \"cloudy\"}"}], tools=TOOLS),
    dict(messages=[U, {"role": "assistant", "content": "", "tool_calls": [call("a", "get_weather", {"city": "Toronto"}),
                                                                          call("b", "get_weather", {"city": "Paris", "days": 2})]},
                   {"role": "tool", "tool_call_id": "a", "content": "3C, cloudy"},
                   {"role": "tool", "tool_call_id": "b", "content": "11C, sunny"}], tools=TOOLS),
    dict(messages=[U, {"role": "assistant", "content": "", "tool_calls": [call("c1", "get_weather", {"city": "Toronto"})]},
                   {"role": "tool", "tool_call_id": "c1", "content": "3C, cloudy"},
                   {"role": "assistant", "content": "It's 3C and cloudy in Toronto."},
                   {"role": "user", "content": "And files with 'TODO'?"},
                   {"role": "assistant", "content": "", "tool_calls": [call("c2", "search_files", {"limit": 5, "options": {"case": True, "depth": 2},
                                                                                                 "paths": ["src", "tools"], "query": "TODO", "regex": None})]},
                   {"role": "tool", "tool_call_id": "c2", "content": "src/model.cpp:12: TODO"}], tools=TOOLS),
    dict(messages=[U, {"role": "assistant", "content": "", "tool_calls": [call("t", "get_time", {})]},
                   {"role": "tool", "tool_call_id": "t", "content": "18:02"}], tools=TOOLS),
]

def render(tmpl, case, think):
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
    t = env.from_string(tmpl)
    return t.render(messages=case["messages"], tools=case.get("tools"), add_generation_prompt=True, enable_thinking=think,
                    bos_token="<bos>")

exe, paths = sys.argv[1], sys.argv[2:]
ok = True
for path in paths:
    tmpl = load_template(path)
    empty_thought = "<|channel>thought\\n<channel|>" in tmpl
    inputs, wants = [], []
    for case in CASES:
        for think in (False, True):
            inputs.append({"messages": case.get("tierlm", case["messages"]), "tools": case.get("tools", []), "think": int(think), "empty_thought": empty_thought})
            wants.append(render(tmpl, case, think))
    r = subprocess.run([exe, "gemma4-prompt"], input=json.dumps(inputs), capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0: sys.exit(r.stderr)
    gots = r.stdout.split("\x01")[:-1]
    n_ok = 0
    for i, (got, want) in enumerate(zip(gots, wants)):
        if got == want: n_ok += 1; continue
        ok = False
        k = next((j for j in range(min(len(got), len(want))) if got[j] != want[j]), min(len(got), len(want)))
        print(f"FAIL case {i // 2} thinking {'on' if i % 2 else 'off'}: differs at {k}\n  want: {want[max(0, k - 80):k + 120]!r}\n  got:  {got[max(0, k - 80):k + 120]!r}")
    print(f"{path.split('/')[-1].split(chr(92))[-1]}: {n_ok} of {len(wants)} prompts match the template")

# Reading calls back out of replies as the model writes them (the reply as tierlm shows it: thinking as <think>).
Q = '<|"|>'
replies = [
    ({"raw": f"<|tool_call>call:get_weather{{city:{Q}Toronto{Q},unit:{Q}c{Q}}}<tool_call|>", "tools": TOOLS},
     "", [("get_weather", {"city": "Toronto", "unit": "c"})]),
    ({"raw": f"Let me check.<|tool_call>call:get_weather{{city:{Q}Paris, France{Q},days:{Q}3{Q}}}<tool_call|><|tool_call>call:get_time{{}}<tool_call|>", "tools": TOOLS},
     "Let me check.", [("get_weather", {"city": "Paris, France", "days": 3}), ("get_time", {})]),
    ({"raw": f"<|channel>thought\nNeed files.<channel|><|tool_call>call:search_files{{query:{Q}a, b: {{c}}{Q},paths:[{Q}src{Q},{Q}x y{Q}],limit:2.5,regex:false,"
             f"options:{{case:true,depth:3}},filters:[{{ext:{Q}.cpp{Q}}}]}}<tool_call|>", "tools": TOOLS},
     "<think>Need files.</think>", [("search_files", {"query": "a, b: {c}", "paths": ["src", "x y"], "limit": 2.5, "regex": False,
                                                       "options": {"case": True, "depth": 3}, "filters": [{"ext": ".cpp"}]})]),
    ({"raw": "No tools needed: it's 3C.", "tools": TOOLS}, "No tools needed: it's 3C.", []),
]
r = subprocess.run([exe, "gemma4-reply"], input=json.dumps([x[0] for x in replies]), capture_output=True, text=True, encoding="utf-8")
if r.returncode != 0: sys.exit(r.stderr)
n_ok = 0
for (inp, want_text, want_calls), line in zip(replies, r.stdout.splitlines()):
    got = json.loads(line)
    calls = [(c["name"], c["arguments"]) for c in got["calls"]]
    if got["display"].strip() == want_text and calls == want_calls: n_ok += 1; continue
    ok = False
    print(f"FAIL reply {inp['raw'][:60]!r}\n  want: {want_text!r} {want_calls}\n  got:  {got['display']!r} {calls}")
print(f"replies: {n_ok} of {len(replies)} read correctly")
sys.exit(0 if ok else 1)
