"""GLM's chat format and tool calls against its own chat templates.

Renders conversations (thinking on and off, system messages, earlier replies with their thinking before and after the
last user message, tools, tool calls with text and other values, several calls at once and their results, the answer
after them) with Z.ai's templates (the copies llama.cpp keeps: tests/glm-4.6.jinja for 4.5 / 4.6, tests/glm-4.7-flash.jinja; a
GGUF's own can be given instead) through transformers' Jinja renderer, and checks that tierlm builds the same prompt, character for
character. Then reads replies back: thinking, text and tool calls in either layout.

usage: python tests/glm_chat_test.py build/tierlm-chat-check [template.jinja | model.gguf ...]
       (by default both: tests/glm-4.6.jinja and tests/glm-4.7-flash.jinja)
"""
import json, os, subprocess, sys
from transformers.utils.chat_template_utils import render_jinja_template

def load_template(path):
    if path.endswith(".jinja"): return open(path, encoding="utf-8").read()
    import gguf
    return gguf.GGUFReader(path).fields["tokenizer.chat_template"].contents()

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city, café ☕.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string", "description": "The city, e.g. Toronto"},
        "unit": {"type": "string", "enum": ["c", "f"]}, "days": {"type": "integer"},
        "hourly": {"type": "boolean"}, "spots": {"type": "array", "items": {"type": "string"}}}, "required": ["city"]}}}
NOARGS = {"type": "function", "function": {"name": "get_time", "description": "The current time.",
                                           "parameters": {"type": "object", "properties": {}}}}
TOOLS = [WEATHER, NOARGS]
def call(id_, name, args): return {"id": id_, "type": "function", "function": {"name": name, "arguments": args}}
U = {"role": "user", "content": "What's the weather in Toronto?"}
CASES = [
    dict(messages=[{"role": "user", "content": "Hello!"}]),
    dict(messages=[{"role": "user", "content": "Be quick /nothink"}]),
    dict(messages=[{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"},
                   {"role": "assistant", "reasoning_content": "The user greets.", "content": "Hello! How can I help?"},
                   {"role": "user", "content": "Tell me a joke, café-style ☕."}],
         tierlm=[{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"},
                 {"role": "assistant", "content": "<think>The user greets.</think>\n\nHello! How can I help?"},
                 {"role": "user", "content": "Tell me a joke, café-style ☕."}]),
    dict(messages=[U], tools=TOOLS),
    dict(messages=[{"role": "system", "content": "You are a coding agent."}, U], tools=TOOLS),
    # after the last user message: the reply's thinking stays
    dict(messages=[U, {"role": "assistant", "reasoning_content": "Need the weather.", "content": "",
                       "tool_calls": [call("c1", "get_weather", {"unit": "c", "city": "Toronto", "days": 2, "hourly": True,
                                                                 "spots": ["CN Tower", "Harbourfront"]})]},
                   {"role": "tool", "tool_call_id": "c1", "content": "{\"temp\": 3, \"sky\": \"cloudy\"}"}], tools=TOOLS,
         tierlm=[U, {"role": "assistant", "content": "<think>Need the weather.</think>\n\n",
                     "tool_calls": [call("c1", "get_weather", {"unit": "c", "city": "Toronto", "days": 2, "hourly": True,
                                                               "spots": ["CN Tower", "Harbourfront"]})]},
                 {"role": "tool", "tool_call_id": "c1", "content": "{\"temp\": 3, \"sky\": \"cloudy\"}"}]),
    dict(messages=[U, {"role": "assistant", "content": "Checking both.", "tool_calls": [call("a", "get_weather", {"city": "Toronto"}),
                                                                                        call("b", "get_weather", {"city": "Paris", "days": 2})]},
                   {"role": "tool", "tool_call_id": "a", "content": "3C"}, {"role": "tool", "tool_call_id": "b", "content": "11C"},
                   {"role": "assistant", "content": "Toronto 3C, Paris 11C."}, {"role": "user", "content": "Time?"},
                   {"role": "assistant", "content": "", "tool_calls": [call("t", "get_time", {})]},
                   {"role": "tool", "tool_call_id": "t", "content": "12:00"}], tools=TOOLS),
]
WT = [WEATHER, NOARGS]
REPLIES = [   # (raw reply as tierlm keeps it, display, calls)
    ("Hello there.", "Hello there.", []),
    ("<think>Greeting.</think>Hi!", "<think>Greeting.</think>Hi!", []),
    ("<think>Need weather.</think>Let me check.<tool_call>get_weather<arg_key>city</arg_key><arg_value>Toronto</arg_value>"
     "<arg_key>days</arg_key><arg_value>2</arg_value><arg_key>hourly</arg_key><arg_value>true</arg_value></tool_call>",
     "<think>Need weather.</think>Let me check.", [("get_weather", {"city": "Toronto", "days": 2, "hourly": True})]),
    ("\n<think></think>\n<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>New York</arg_value>\n"
     "<arg_key>spots</arg_key>\n<arg_value>[\"MoMA\", \"Pier 17\"]</arg_value>\n</tool_call>\n<tool_call>get_time\n</tool_call>",
     "\n<think></think>", [("get_weather", {"city": "New York", "spots": ["MoMA", "Pier 17"]}), ("get_time", {})]),
    ("<tool_call>get_weather<arg_key>city</arg_key><arg_value>123</arg_value></tool_call>", "",
     [("get_weather", {"city": "123"})]),
    ("Wrap a call in <tool_call> tags like this.</tool_call>", "Wrap a call in", []),   # (prose, not a call)
    ("<tool_call>delete_files<arg_key>path</arg_key><arg_value>/</arg_value></tool_call>", "", []),   # (not one of the tools)
]

# As an app sends a conversation: a tool call's arguments as a JSON string (its keys in the order written)
def as_sent(msgs):
    out = []
    for m in msgs:
        m = dict(m)
        if "tool_calls" in m:
            m["tool_calls"] = [{**tc, "function": {**tc["function"], "arguments": json.dumps(tc["function"]["arguments"], ensure_ascii=False)}}
                               for tc in m["tool_calls"]]
        out.append(m)
    return out

exe = sys.argv[1]
here = os.path.dirname(__file__)
paths = sys.argv[2:] or [os.path.join(here, "glm-4.6.jinja"), os.path.join(here, "glm-4.7-flash.jinja")]
ok = True
total = 0
for path in paths:
    tmpl = load_template(path)
    layout = "46" if "/nothink" in tmpl else "47"
    inputs, wants = [], []
    for c in CASES:
        for think in (0, 1):
            kw = {} if think else {"enable_thinking": False}
            wants.append(render_jinja_template(conversations=[c["messages"]], tools=c.get("tools"), chat_template=tmpl,
                                               add_generation_prompt=True, **kw)[0][0])
            inputs.append({"messages": as_sent(c.get("tierlm", c["messages"])), "tools": c.get("tools", []), "think": think, "layout": layout})
    r = subprocess.run([exe, "glm-prompt"], input=json.dumps(inputs), capture_output=True, text=True, encoding="utf-8")
    if r.returncode: sys.exit(r.stderr)
    got = r.stdout.split("\x01")[:-1]
    bad = 0
    for i, (g, w) in enumerate(zip(got, wants)):
        if g != w:
            bad += 1
            k = next((j for j in range(min(len(g), len(w))) if g[j] != w[j]), min(len(g), len(w)))
            print(f"{os.path.basename(path)} prompt {i}: differs at {k}:\n  tierlm:   {g[max(0, k - 80):k + 80]!r}\n  template: {w[max(0, k - 80):k + 80]!r}")
    ok &= bad == 0 and len(got) == len(wants)
    total += len(got)
    print(f"{os.path.basename(path)} (layout {layout}): {len(got)} prompts, {'all match' if bad == 0 else f'{bad} MISMATCH'}")
r = subprocess.run([exe, "glm-reply"], input=json.dumps([{"raw": x[0], "tools": WT} for x in REPLIES]), capture_output=True, text=True, encoding="utf-8")
if r.returncode: sys.exit(r.stderr)
for line, (raw, disp, calls) in zip(r.stdout.splitlines(), REPLIES):
    j = json.loads(line)
    gc = [(c["name"], c["arguments"]) for c in j["calls"]]
    if j["display"] != disp or gc != calls:
        ok = False
        print(f"reply {raw[:60]!r}: display {j['display']!r} (want {disp!r}), calls {gc} (want {calls})")
print("replies:", "ok" if ok else "FAIL")
sys.exit(0 if ok else 1)
