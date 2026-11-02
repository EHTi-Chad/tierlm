"""Checks tierlm's harmony chat format (gpt-oss) against OpenAI's own chat template.

The prompt tierlm builds must be byte for byte what the template (tests/openai-gpt-oss.jinja, from llama.cpp's copy of
OpenAI's gpt-oss template) gives when run through Jinja the way Hugging Face runs chat templates; and replies written
the way gpt-oss writes them must come out as the right text and tool calls.

usage: python tests/harmony_test.py build/tierlm-chat-check        (needs: pip install jinja2)
"""
import json, os, subprocess, sys
import jinja2

DATE = "2026-11-01"
tmpl = open(os.path.join(os.path.dirname(__file__), "openai-gpt-oss.jinja"), encoding="utf-8").read()
env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
env.filters["tojson"] = lambda x, ensure_ascii=False, indent=None, separators=None, sort_keys=False: \
    json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
def raise_exception(m): raise Exception(m)
T = env.from_string(tmpl)

tools = [
    {"type": "function", "function": {"name": "get_weather", "description": "Get the weather", "parameters": {"type": "object",
     "properties": {"city": {"type": "string", "description": "City name"}, "unit": {"type": "string", "enum": ["c", "f"], "default": "c"},
                    "days": {"type": "integer", "default": 3}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "todowrite", "description": "Write the todo list", "parameters": {"type": "object",
     "properties": {"todos": {"type": "array", "description": "The list", "items": {"type": "object", "properties": {
         "content": {"type": "string"}, "status": {"type": "string", "enum": ["pending", "done"]}, "id": {"type": "string"}},
         "required": ["content", "status"]}}}, "required": ["todos"]}}},
    {"type": "function", "function": {"name": "read", "description": "Read a file", "parameters": {"type": "object", "properties": {
        "filePath": {"type": "string", "description": "Absolute path"}, "offset": {"type": "number"}, "limit": {"type": "number"},
        "tags": {"type": "array", "items": {"type": "string"}}, "flag": {"type": "boolean", "nullable": True},
        "meta": {"type": "object", "properties": {"a": {"type": "string"}}}, "mixed": {"type": ["string", "number"]}}, "required": ["filePath"]}}},
    {"type": "function", "function": {"name": "noop", "description": "Does nothing", "parameters": {"type": "object", "properties": {}}}},
]
call = lambda name, args: {"type": "function", "function": {"name": name, "arguments": args}}
cases = [
    {"messages": [{"role": "user", "content": "Hi there!"}]},
    {"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hello"}]},
    {"messages": [{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "A1"}, {"role": "user", "content": "Q2"}]},
    {"messages": [{"role": "user", "content": "Weather in Paris?"}], "tools": tools},
    {"messages": [{"role": "system", "content": "You are a coder."}, {"role": "user", "content": "Weather?"},
                  {"role": "assistant", "content": "", "thinking": "Need the tool.", "tool_calls": [call("get_weather", {"city": "Paris", "unit": "c"})]},
                  {"role": "tool", "content": "{\"temp\": 21, \"note\": \"sunny ☀\"}"}], "tools": tools},
    {"messages": [{"role": "user", "content": "Weather?"},
                  {"role": "assistant", "thinking": "Need the tool.", "tool_calls": [call("get_weather", {"city": "Paris"})]},
                  {"role": "tool", "content": "21C"}, {"role": "assistant", "content": "It's 21C."}, {"role": "user", "content": "Thanks"}], "tools": tools},
]
ok = True
out = subprocess.run([sys.argv[1], "prompt"], input=json.dumps(cases).encode(), capture_output=True, env={**os.environ, "TIERLM_DATE": DATE})
got = out.stdout.decode("utf-8").split("\x01")
same = 0
for i, c in enumerate(cases):
    want = T.render(messages=c["messages"], tools=c.get("tools"), add_generation_prompt=True, strftime_now=lambda f: DATE,
                    raise_exception=raise_exception)
    if got[i] == want: same += 1
    else:
        k = next((j for j in range(min(len(got[i]), len(want))) if got[i][j] != want[j]), min(len(got[i]), len(want)))
        print(f"  prompt {i} differs at {k}:\n    tierlm:   {got[i][max(0, k - 60):k + 60]!r}\n    template: {want[max(0, k - 60):k + 60]!r}")
ok &= same == len(cases)
print(f"prompts   : {same} of {len(cases)} identical to OpenAI's template  ->  {'ok' if same == len(cases) else 'FAIL'}")

replies = [  # (raw reply as gpt-oss writes it, expected display, expected calls)
    ("<|channel|>analysis<|message|>User asks for 2+2.<|end|><|start|>assistant<|channel|>final<|message|>2 + 2 = 4.",
     "<think>User asks for 2+2.</think>\n\n2 + 2 = 4.", []),
    ("<|channel|>analysis<|message|>Need weather.<|end|><|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"city\":\"Paris\"}",
     "<think>Need weather.</think>\n\n", [("get_weather", {"city": "Paris"})]),
    ("<|channel|>analysis<|message|>x<|end|><|start|>assistant to=functions.read<|channel|>commentary json<|message|>{\"filePath\": \"/a/b\"}",
     "<think>x</think>\n\n", [("read", {"filePath": "/a/b"})]),
    ("<|channel|>commentary<|message|>Let me check.<|end|><|start|>assistant<|channel|>final<|message|>Done", "Let me check.\n\nDone", []),
    ("<|channel|>analy", "", []),
    ("Just text", "Just text", []),
]
out = subprocess.run([sys.argv[1], "reply"], input=json.dumps([r[0] for r in replies]).encode(), capture_output=True)
good = 0
for (raw, disp, calls), line in zip(replies, out.stdout.decode("utf-8").splitlines()):
    j = json.loads(line)
    if j["display"] == disp and [(c["name"], c["arguments"]) for c in j["calls"]] == calls: good += 1
    else: print(f"  reply {raw!r}\n    got {j}")
ok &= good == len(replies)
print(f"replies   : {good} of {len(replies)} read correctly  ->  {'ok' if good == len(replies) else 'FAIL'}")
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
