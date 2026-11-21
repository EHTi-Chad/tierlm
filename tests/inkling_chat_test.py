"""Inkling 2's chat format and tool calls against its own chat template.

Renders conversations (thinking on and off, system messages, earlier replies with their thinking, tools, tool calls
and their results, several calls at once, the answer after the results) with Thinking Machines' chat template (the copy
llama.cpp keeps, tests/inkling2-small.jinja; a GGUF's own can be given instead) through transformers' Jinja renderer,
and checks that tierlm builds the same prompt, character for character. Then reads replies written in Inkling's blocks
back: thinking, text and tool calls.

usage: python tests/inkling_chat_test.py build/tierlm-chat-check [template.jinja | model.gguf]
"""
import json, os, subprocess, sys
from transformers.utils.chat_template_utils import render_jinja_template

def load_template(path):
    if path.endswith(".jinja"): return open(path, encoding="utf-8").read()
    import gguf
    return gguf.GGUFReader(path).fields["tokenizer.chat_template"].contents()

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string", "description": "The city, e.g. Toronto"},
        "unit": {"type": "string", "enum": ["c", "f"]}, "days": {"type": "integer"}}, "required": ["city"]}}}
NOARGS = {"type": "function", "function": {"name": "get_time", "description": "The current time.",
                                           "parameters": {"type": "object", "properties": {}}}}
TOOLS = [WEATHER, NOARGS]
def call(id_, name, args): return {"id": id_, "type": "function", "function": {"name": name, "arguments": args}}
U = {"role": "user", "content": "What's the weather in Toronto?"}
CASES = [
    dict(messages=[{"role": "user", "content": "Hello!"}]),
    dict(messages=[{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"},
                   {"role": "assistant", "reasoning_content": "The user greets.", "content": "Hello! How can I help?"},
                   {"role": "user", "content": "Tell me a joke, café-style ☕."}],
         tierlm=[{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"},
                 {"role": "assistant", "content": "<think>The user greets.</think>\n\nHello! How can I help?"},
                 {"role": "user", "content": "Tell me a joke, café-style ☕."}]),
    dict(messages=[U], tools=TOOLS),
    dict(messages=[{"role": "system", "content": "You are a coding agent."}, U], tools=TOOLS),
    dict(messages=[U, {"role": "assistant", "content": "", "tool_calls": [call("c1", "get_weather", {"unit": "c", "city": "Toronto"})]},
                   {"role": "tool", "tool_call_id": "c1", "content": "{\"temp\": 3, \"sky\": \"cloudy\"}"}], tools=TOOLS),
    dict(messages=[U, {"role": "assistant", "content": "", "tool_calls": [call("a", "get_weather", {"city": "Toronto"}),
                                                                          call("b", "get_weather", {"city": "Paris", "days": 2})]},
                   {"role": "tool", "tool_call_id": "a", "content": "3C"}, {"role": "tool", "tool_call_id": "b", "content": "11C"},
                   {"role": "assistant", "content": "Toronto 3C, Paris 11C."}, {"role": "user", "content": "Time?"},
                   {"role": "assistant", "content": "", "tool_calls": [call("t", "get_time", {})]},
                   {"role": "tool", "tool_call_id": "t", "content": "12:00"}], tools=TOOLS),
]
REPLIES = [
    ("<|message_model|><|content_text|>Hello there.<|end_message|><|content_model_end_sampling|>", "Hello there.", []),
    ("<|message_model|><|content_thinking|>Greeting.<|end_message|><|message_model|><|content_text|>Hi!<|end_message|>",
     "<think>Greeting.</think>\n\nHi!", []),
    ("<|message_model|><|content_thinking|>Need weather.<|end_message|><|message_model|>get_weather<|content_invoke_tool_json|>"
     "{\"name\":\"get_weather\",\"args\":{\"city\":\"Toronto\",\"days\":2}}<|end_message|>",
     "<think>Need weather.</think>\n\n", [("get_weather", {"city": "Toronto", "days": 2})]),
    ("<|message_model|><|content_text|>Partly writ", "Partly writ", []),
    ("<|message_model|><|content_think", "", []),
]

exe = sys.argv[1]
tmpl = load_template(sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "inkling2-small.jinja"))
ok = True
inputs, wants = [], []
for c in CASES:
    for think in (0, 1):
        kw = {} if think else {"reasoning_effort": 0}
        wants.append(render_jinja_template(conversations=[c["messages"]], tools=c.get("tools"), chat_template=tmpl, **kw)[0][0])
        inputs.append({"messages": c.get("tierlm", c["messages"]), "tools": c.get("tools", []), "think": think})
r = subprocess.run([exe, "inkling-prompt"], input=json.dumps(inputs), capture_output=True, text=True, encoding="utf-8")
if r.returncode: sys.exit(r.stderr)
got = r.stdout.split("\x01")[:-1]
for i, (g, w) in enumerate(zip(got, wants)):
    if g != w:
        ok = False
        k = next((j for j in range(min(len(g), len(w))) if g[j] != w[j]), min(len(g), len(w)))
        print(f"prompt {i}: differs at {k}:\n  tierlm:   {g[max(0, k - 80):k + 80]!r}\n  template: {w[max(0, k - 80):k + 80]!r}")
print(f"prompts: {len(got) - (0 if ok else 1)} checked, {'all match' if ok else 'MISMATCH'}")
r = subprocess.run([exe, "inkling-reply"], input=json.dumps([x[0] for x in REPLIES]), capture_output=True, text=True, encoding="utf-8")
for line, (raw, disp, calls) in zip(r.stdout.splitlines(), REPLIES):
    j = json.loads(line)
    gc = [(c["name"], c["arguments"]) for c in j["calls"]]
    if j["display"] != disp or gc != calls:
        ok = False
        print(f"reply {raw[:60]!r}: display {j['display']!r} (want {disp!r}), calls {gc} (want {calls})")
print("replies:", "ok" if ok else "FAIL")
sys.exit(0 if ok else 1)
