"""Qwen3.5 / 3.6 / 3.8 (and Flash-Next) conversations against the models' own chat templates.

Renders conversations as apps send them (system messages, text with whitespace around it, content in parts, earlier
replies with and without their thinking, tools, tool calls with every kind of argument, their results, several calls
at once) with the templates from the GGUF files through transformers' Jinja renderer, thinking off and on and at
each reasoning effort, and checks that tierlm builds the same prompt, character for character:
  tests/qwen3.8-unsloth.jinja  Qwen3.8-27B and Qwen3.8 Flash-Next, Unsloth's files (its fixed template)
  tests/qwen3.8.jinja          the same models, Qwen's own template
  tests/qwen3.6.jinja          Qwen3.6-35B-A3B (with preserve_thinking, which is how tierlm writes earlier replies:
                               as the model wrote them, so a cache that can't rewind carries on)
Where a template refuses a conversation that tierlm accepts (Qwen's own: a "developer" message, two system messages,
effort "high"), tierlm's prompt is checked against Unsloth's fixed template's instead.

usage: python tests/qwen_chat_test.py build/tierlm-chat-check
"""
import json, os, subprocess, sys
from transformers.utils.chat_template_utils import render_jinja_template

here = os.path.dirname(os.path.abspath(__file__))
exe = sys.argv[1] if len(sys.argv) > 1 else "build/tierlm-chat-check"
def tpl(name): return open(os.path.join(here, name), encoding="utf-8").read()

TOOLS = [{"type": "function", "function": {"name": "bash", "description": "Execute a bash command", "parameters": {"type": "object", "properties": {
              "command": {"type": "string", "description": "Bash command"}, "timeout": {"type": "number"}}, "required": ["command"]}}},
         {"type": "function", "function": {"name": "read", "description": "Read a file <or> an image & more: \"quoted\", café", "parameters": {"type": "object", "properties": {
              "path": {"type": "string"}, "offset": {"type": "number"}}, "required": ["path"]}}}]
def call(name, args, id="c1", as_string=True):
    return {"id": id, "type": "function", "function": {"name": name, "arguments": json.dumps(args) if as_string else args}}
SYS = "You are an expert coding assistant.\n\nGuidelines:\n- Be concise\n"
PIC = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
CASES = {
 "plain": dict(messages=[{"role": "user", "content": "hi"}]),
 "system, whitespace around the text": dict(messages=[{"role": "system", "content": SYS}, {"role": "user", "content": "  hi there \n"}]),
 "unicode whitespace": dict(messages=[{"role": "system", "content": " Be brief.　"}, {"role": "user", "content": " hi "}]),
 "two turns": dict(messages=[{"role": "system", "content": SYS}, {"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hello!\n"},
                             {"role": "user", "content": "more"}]),
 "tools, first turn": dict(messages=[{"role": "system", "content": SYS}, {"role": "user", "content": "list files"}], tools=TOOLS),
 "tools, no system": dict(messages=[{"role": "user", "content": "list files"}], tools=TOOLS),
 "tool loop": dict(messages=[{"role": "system", "content": SYS}, {"role": "user", "content": "list files"},
                             {"role": "assistant", "content": "Let me look.", "tool_calls": [call("bash", {"command": "ls -la", "timeout": 5})]},
                             {"role": "tool", "tool_call_id": "c1", "content": "a.txt\nb.txt\n"}], tools=TOOLS),
 "tool loop, no text": dict(messages=[{"role": "system", "content": SYS}, {"role": "user", "content": "list files"},
                             {"role": "assistant", "content": None, "tool_calls": [call("bash", {"command": "echo '<x> & \"q\"'\necho two"})]},
                             {"role": "tool", "tool_call_id": "c1", "content": "<x> & \"q\"\ntwo"},
                             {"role": "assistant", "content": "", "tool_calls": [call("read", {"path": "a.txt", "offset": 3}, "c2")]},
                             {"role": "tool", "tool_call_id": "c2", "content": "  line\n\n"}], tools=TOOLS),
 "several calls at once": dict(messages=[{"role": "user", "content": "both"},
                             {"role": "assistant", "content": "Both:", "tool_calls": [call("bash", {"command": "ls"}), call("read", {"path": "x"}, "c2")]},
                             {"role": "tool", "tool_call_id": "c1", "content": "one"}, {"role": "tool", "tool_call_id": "c2", "content": "two"},
                             {"role": "assistant", "content": "Done."}, {"role": "user", "content": "thanks, again"}], tools=TOOLS),
 "thinking in earlier replies": dict(messages=[{"role": "system", "content": SYS}, {"role": "user", "content": "list files"},
                             {"role": "assistant", "content": "Let me look.", "reasoning_content": "I should run ls.", "tool_calls": [call("bash", {"command": "ls"})]},
                             {"role": "tool", "tool_call_id": "c1", "content": "a.txt"},
                             {"role": "assistant", "content": "There is a.txt.", "reasoning_content": "\nOnly one file.\n"},
                             {"role": "user", "content": "and now?"},
                             {"role": "assistant", "content": "", "reasoning_content": "Again.", "tool_calls": [call("bash", {"command": "ls"}, "c3")]},
                             {"role": "tool", "tool_call_id": "c3", "content": "a.txt\nb.txt"}], tools=TOOLS),
 "every kind of argument": dict(messages=[{"role": "user", "content": "go"},
                             {"role": "assistant", "content": "", "tool_calls": [call("bash", {"command": "x", "opts": {"a": [1, 2, {"b": None}], "c": True, "d": 1.5, "e": "é <&> '"},
                                                                                              "n": 7, "flag": False, "list": ["p", "q"]})]},
                             {"role": "tool", "tool_call_id": "c1", "content": "ok"}], tools=TOOLS),
 "content in parts, an image": dict(messages=[{"role": "system", "content": [{"type": "text", "text": SYS}]},
                             {"role": "user", "content": [{"type": "text", "text": "part one "}, PIC, {"type": "text", "text": " part two"}]}]),
 "developer role": dict(messages=[{"role": "developer", "content": "Dev rules."}, {"role": "user", "content": "hi"}], tools=TOOLS),
 "two system messages": dict(messages=[{"role": "system", "content": "First.\n"}, {"role": "system", "content": " Second."}, {"role": "user", "content": "hi"}]),
}
# (what tierlm is told, what the template is told)
MODES = {"off": (dict(think=0), dict(enable_thinking=False)),
         "on": (dict(think=1), dict(enable_thinking=True)),
         "on, medium": (dict(think=1, effort="medium"), dict(enable_thinking=True, reasoning_effort="medium")),
         "on, low": (dict(think=1, effort="low"), dict(enable_thinking=True, reasoning_effort="low")),
         "on, high": (dict(think=1, effort="high"), dict(enable_thinking=True, reasoning_effort="high")),
         "on, xhigh": (dict(think=1, effort="xhigh"), dict(enable_thinking=True, reasoning_effort="xhigh"))}
TEMPLATES = {"qwen3.8-unsloth.jinja": dict(effort=True, kw={}), "qwen3.8.jinja": dict(effort=True, kw={}),
             "qwen3.6.jinja": dict(effort=False, kw={"preserve_thinking": True})}

def for_template(msgs):          # as a pipeline hands them to a template: tool-call arguments as objects, no null text
    out = []
    for m in msgs:
        m = dict(m)
        if m.get("tool_calls"):
            m["tool_calls"] = [{**tc, "function": {**tc["function"], "arguments": json.loads(tc["function"]["arguments"])
                                if isinstance(tc["function"]["arguments"], str) else tc["function"]["arguments"]}} for tc in m["tool_calls"]]
        if m.get("content") is None: m["content"] = ""
        out.append(m)
    return out
def render(name, c, kw):
    t = TEMPLATES[name]
    if not t["effort"]: kw = {k: v for k, v in kw.items() if k != "reasoning_effort"}
    return render_jinja_template(conversations=[for_template(c["messages"])], tools=c.get("tools"), chat_template=tpl(name),
                                 add_generation_prompt=True, **t["kw"], **kw)[0][0]

fails, total, refused = [], 0, 0
for name, t in TEMPLATES.items():
    jobs, wants = [], []
    for cname, c in CASES.items():
        for mname, (ours, theirs) in MODES.items():
            try: want = render(name, c, theirs)
            except Exception:                    # the template refuses it: as Unsloth's fixed template writes it
                refused += 1
                if not t["effort"]: continue
                want = render("qwen3.8-unsloth.jinja", c, theirs)
            jobs.append({"messages": c["messages"], "tools": c.get("tools", []), "effort_lines": t["effort"], **ours})
            wants.append((f"{name}: {cname} / {mname}", want))
    r = subprocess.run([exe, "qwen-prompt"], input=json.dumps(jobs).encode(), capture_output=True)
    if r.returncode: sys.exit(f"tierlm-chat-check failed: {r.stderr.decode()[-500:]}")
    got = r.stdout.decode("utf-8").split("\x01")[:-1]
    assert len(got) == len(wants)
    for (label, want), g in zip(wants, got):
        total += 1
        if g != want:
            fails.append(label)
            if len(fails) <= 3:
                i = next((k for k in range(min(len(g), len(want))) if g[k] != want[k]), min(len(g), len(want)))
                print(f"DIFFERENT: {label}\n  template: ...{want[max(0, i - 60):i + 60]!r}\n  tierlm:   ...{g[max(0, i - 60):i + 60]!r}")
print(f"{total} prompts checked against {len(TEMPLATES)} templates ({refused} conversations a template refuses), {len(fails)} different")
print("PASS" if not fails else "FAIL: " + "; ".join(fails[:10]))
sys.exit(1 if fails else 0)
