#!/usr/bin/env python
"""Drive Unsloth Studio's own proxy code against qwfn-server: ExternalProviderClient
(the httpx client with the 300 s read timeout, the SSE relay and sanitizer) for a chat
turn, and stream_with_studio_tools + OAICompatTransport (Unsloth's tool loop, executing
its python/terminal/edit_file tools in its sandbox) for a tools turn -- the exact code the
Studio route runs for the "Custom" provider once the UI is logged in. Records every SSE
line the UI would receive, with arrival times."""
import argparse, asyncio, base64, json, os, sys, threading, time
BACKEND = os.path.expanduser("~/.unsloth/studio/unsloth_studio/lib/python3.13/site-packages/studio/backend")
sys.path.insert(0, BACKEND)
from core.inference.external_provider import ExternalProviderClient
from core.inference.external_tool_transport import OAICompatTransport
from core.inference import studio_tool_loop as loop_mod
from core.inference.studio_tool_loop import stream_with_studio_tools, ToolLoopRun, ToolLoopPolicy
from core.inference.tools import ALL_TOOLS

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["chat", "tools"], default="chat")
ap.add_argument("--prompt", required=True)
ap.add_argument("--image", action="append", default=[])
ap.add_argument("--image-first", action="store_true", help="put the image part before the text part")
ap.add_argument("--effort", default=None, help="reasoning_effort to hand Studio's client (it decides what to forward)")
ap.add_argument("--thinking", default=None, help="enable_thinking true/false")
ap.add_argument("--max-tokens", type=int, default=126000)
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--tools", default="python,terminal,edit_file,render_html")
ap.add_argument("--session", default="__LOCALID_claudetest")
ap.add_argument("--model", default="qwen3.8-flash-next")
ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
ap.add_argument("--out", required=True, help="log file")
a = ap.parse_args()
a.image = [os.path.abspath(p) for p in a.image]; a.out = os.path.abspath(a.out)
os.chdir(BACKEND)
LOG = open(a.out, "a")
T0 = time.time()
def log(s):
    line = "[%7.1f s] %s" % (time.time() - T0, s)
    print(line, flush=True); LOG.write(line + "\n"); LOG.flush()

def data_url(p):
    ext = p.rsplit(".", 1)[-1].lower()
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "gif": "image/gif", "bmp": "image/bmp"}[ext]
    return f"data:{mime};base64," + base64.b64encode(open(p, "rb").read()).decode()

text_part = {"type": "text", "text": a.prompt}
img_parts = [{"type": "image_url", "image_url": {"url": data_url(p)}} for p in a.image]
if a.image:
    content = img_parts + [text_part] if a.image_first else [text_part] + img_parts
else:
    content = a.prompt
messages = [{"role": "user", "content": content}]
log(f"mode={a.mode} images={len(a.image)} image_first={a.image_first} effort={a.effort} thinking={a.thinking} max_tokens={a.max_tokens} prompt={a.prompt[:80]!r}")

client = ExternalProviderClient(provider_type="custom", base_url=a.base_url, api_key="")
kw = dict(temperature=a.temperature, top_p=0.95, max_tokens=a.max_tokens, presence_penalty=0.0)
if a.effort is not None: kw["reasoning_effort"] = a.effort
if a.thinking is not None: kw["enable_thinking"] = a.thinking.lower() == "true"

class Acc:
    content = ""; reasoning = ""; lines = 0; data = 0; comments = 0; errors = []; events = []; calls = {}; finish = None; usage = None; timings = None; gaps = []

async def consume(gen, acc):
    last = time.time()
    async for line in gen:
        now = time.time(); gap = now - last; last = now
        acc.lines += 1
        if gap > 20: acc.gaps.append(round(gap)); log(f"  ({gap:.0f} s between lines)")
        if not line.startswith("data:"):
            acc.comments += 1
            if acc.comments <= 3 or acc.comments % 20 == 0: log(f"  non-data line #{acc.comments}: {line[:60]!r}")
            continue
        raw = line[5:].strip()
        if raw == "[DONE]": log("[DONE]"); continue
        try: p = json.loads(raw)
        except Exception: log(f"  unparseable data: {raw[:120]}"); continue
        acc.data += 1
        if "error" in p: acc.errors.append(p["error"]); log(f"ERROR frame: {json.dumps(p['error'])[:400]}"); continue
        ev = p.get("_toolEvent") or (p if p.get("type") in ("tool_start", "tool_end", "tool_output", "tool_args", "tool_status") else None)
        if ev:
            acc.events.append(ev)
            desc = json.dumps({k: v for k, v in ev.items() if k not in ("result",)})[:220]
            res = ev.get("result"); log(f"toolEvent {desc}" + (f" result[{len(str(res))} chars]: {str(res)[:300]!r}" if res is not None else ""))
        for ch in p.get("choices", []):
            d = ch.get("delta") or {}
            if d.get("reasoning_content"):
                if not acc.reasoning: log("first reasoning delta")
                acc.reasoning += d["reasoning_content"]
            if d.get("content"):
                if not acc.content: log(f"first content delta: {d['content'][:60]!r}")
                acc.content += d["content"]
            for tc in d.get("tool_calls") or []:
                idx = tc.get("index", 0); slot = acc.calls.setdefault(idx, {"name": "", "args": "", "id": None, "frags": 0})
                if tc.get("id"): slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"): slot["name"] = fn["name"]; log(f"tool_calls delta: #{idx} {fn['name']} opened (id {tc.get('id')})")
                if fn.get("arguments"): slot["args"] += fn["arguments"]; slot["frags"] += 1
            if ch.get("finish_reason"): acc.finish = ch["finish_reason"]; log(f"finish_reason={ch['finish_reason']}")
        if p.get("usage"): acc.usage = p["usage"]
        if p.get("timings"): acc.timings = p["timings"]

def summary(acc):
    log("---- summary ----")
    log(f"lines={acc.lines} data={acc.data} comments={acc.comments} gaps>20s={acc.gaps} finish={acc.finish}")
    log(f"usage={acc.usage} timings={json.dumps(acc.timings) if acc.timings else None}")
    log(f"errors={acc.errors}")
    for idx, c in acc.calls.items():
        ok = True
        try: parsed = json.loads(c["args"]); keys = list(parsed) if isinstance(parsed, dict) else type(parsed).__name__
        except Exception as e: ok = False; keys = f"UNPARSEABLE: {e}"
        log(f"call #{idx} {c['name']} id={c['id']} fragments={c['frags']} args={len(c['args'])} chars json_ok={ok} keys={keys}")
    log(f"tool events: {[(e.get('type'), e.get('tool_name')) for e in acc.events]}")
    log(f"reasoning: {len(acc.reasoning)} chars; tail: {acc.reasoning[-300:]!r}")
    log(f"content: {len(acc.content)} chars:\n{acc.content[:4000]}")

async def main():
    acc = Acc()
    try:
        if a.mode == "chat":
            gen = client.stream_chat_completion(messages=messages, model=a.model, stream=True, **kw)
            await consume(gen, acc)
        else:
            names = a.tools.split(",")
            tools = [t for t in ALL_TOOLS if t["function"]["name"] in names]
            log(f"tools: {[t['function']['name'] for t in tools]}")
            loop_mod.build_rag_autoinject = lambda conv, scope: None   # the route's RAG pass reads the app DB; nothing to inject here
            transport = OAICompatTransport(client, model=a.model, continue_final_message=False, enabled_tools=None, stream=True, **kw)
            gen = stream_with_studio_tools(
                transport,
                run=ToolLoopRun(messages=messages, session_id=a.session, thread_id=a.session, model=a.model, tool_choice=None, continue_final_message=False),
                policy=ToolLoopPolicy(tools=tools, max_calls=25, timeout=300, permission_mode="auto", confirm_calls=False,
                                      bypass_permissions=False, rag_scope=None, auto_heal=True, nudge_tool_calls=None),
                cancel_event=threading.Event())
            await consume(gen, acc)
    except Exception as e:
        log(f"EXCEPTION {type(e).__name__}: {e}")
    summary(acc)
asyncio.run(main())
