#!/usr/bin/env python3
"""qwfn-router: one base URL for Claude Code that serves Anthropic's models and the local one.

Claude Code takes a single ANTHROPIC_BASE_URL, and the Claude Desktop app's third-party mode
replaces the provider for a whole profile. This router is the additive way: it listens where
ANTHROPIC_BASE_URL points and forwards each request by the model it names. The local model's
ids go to qwfn-server; everything else goes to api.anthropic.com as it came (headers and body
verbatim, so the claude.ai login, prompt caching and every beta feature are untouched). With a
picker entry for the local model, it shows in /model next to the Anthropic ones, in the same
app and the same list of sessions, and each session chooses.

    python3 tools/qwfn_router.py                    listen on 127.0.0.1:8089
    python3 tools/qwfn_router.py --print-settings   the settings to merge into ~/.claude/settings.json
    python3 tools/qwfn_router.py --configure        merge them (a backup of the file is kept)
    python3 tools/qwfn_router.py --unconfigure      take them out again
    python3 tools/qwfn_router.py --install-service  a systemd user service, so the router is always up
    python3 tools/qwfn_router.py --uninstall-service

    --port N            listen port (8089)
    --server URL        qwfn-server (http://127.0.0.1:8080)
    --upstream URL      where everything else goes (https://api.anthropic.com)
    --local-model ID    a model id to route to the server (repeatable); by default the ids the
                        server reports, plus any id that starts with "qwen" or "local"

Only the Python standard library. One line per request in the log.
"""
import argparse, http.client, http.server, json, os, shutil, socket, subprocess, sys, threading, time, urllib.parse

HOP = {"host", "content-length", "transfer-encoding", "connection", "keep-alive", "proxy-connection",
       "te", "trailer", "upgrade", "accept-encoding"}
RESP_DROP = {"content-length", "transfer-encoding", "connection", "keep-alive", "content-encoding"}

def log(msg):
    sys.stderr.write("[qwfn-router] %s\n" % msg); sys.stderr.flush()

class Router:
    def __init__(self, server, upstream, local_models):
        self.server = urllib.parse.urlsplit(server)
        self.upstream = urllib.parse.urlsplit(upstream)
        self.local_models = set(local_models)
        self._served = set(); self._served_at = 0.0; self._lock = threading.Lock()

    def served_models(self):
        """The ids qwfn-server lists, refreshed once a minute (empty while it is down)."""
        with self._lock:
            if time.time() - self._served_at < 60: return self._served
            self._served_at = time.time()
            try:
                c = self._connect(self.server, timeout=2)
                c.request("GET", "/v1/models"); r = c.getresponse()
                self._served = {m["id"] for m in json.loads(r.read()).get("data", [])}
                c.close()
            except Exception:
                self._served = set()
            return self._served

    def is_local(self, model):
        if not model: return False
        m = model.lower()
        return model in self.local_models or model in self.served_models() or m.startswith(("qwen", "local"))

    @staticmethod
    def _connect(u, timeout):
        if u.scheme == "https":
            return http.client.HTTPSConnection(u.hostname, u.port or 443, timeout=timeout)
        return http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)

class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    router = None   # set at startup

    def log_message(self, *a): pass   # one line per request from proxy() instead

    def do_GET(self): self.proxy()
    def do_POST(self): self.proxy()
    def do_PUT(self): self.proxy()
    def do_DELETE(self): self.proxy()
    def do_PATCH(self): self.proxy()
    def do_OPTIONS(self): self.proxy()
    def do_HEAD(self): self.proxy()

    def fail(self, code, message, kind="api_error"):
        body = json.dumps({"type": "error", "error": {"type": kind, "message": message}}).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers()
        if self.command != "HEAD": self.wfile.write(body)

    def proxy(self):
        R = self.router
        t0 = time.time()
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else b""
        path = self.path
        # Claude Code's connection probe: answered here, it is not for either side.
        if path.split("?")[0] == "/api/hello":
            self.send_response(200); self.send_header("Content-Length", "0"); self.end_headers(); return
        model = ""
        if path.startswith("/v1/messages") and body:
            try: model = str(json.loads(body).get("model") or "")
            except Exception: model = ""
        local = R.is_local(model)
        target = R.server if local else R.upstream
        name = "server" if local else "anthropic"
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
        headers["Host"] = target.netloc
        headers["Accept-Encoding"] = "identity"
        if body: headers["Content-Length"] = str(len(body))
        try:
            c = R._connect(target, timeout=3600)
            c.request(self.command, path, body=body if body else None, headers=headers)
            r = c.getresponse()
        except (OSError, http.client.HTTPException) as e:
            if local:
                self.fail(503, "qwfn-server is not running at %s (%s): start it from the console" % (R.server.geturl(), e))
            else:
                self.fail(502, "cannot reach %s: %s" % (R.upstream.geturl(), e))
            log("%s %s model=%s -> %s: connect failed: %s" % (self.command, path, model or "-", name, e))
            return
        self.send_response(r.status)
        for k, v in r.getheaders():
            if k.lower() not in RESP_DROP: self.send_header(k, v)
        stream = (r.getheader("Content-Type") or "").startswith("text/event-stream")
        sent = 0
        try:
            if self.command == "HEAD":
                self.send_header("Content-Length", "0"); self.end_headers()
            elif stream:
                # Chunk by chunk as it arrives: a delta must not wait for a full buffer.
                self.send_header("Transfer-Encoding", "chunked"); self.end_headers()
                while True:
                    chunk = r.read1(65536)
                    if not chunk: break
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk)); self.wfile.flush(); sent += len(chunk)
                self.wfile.write(b"0\r\n\r\n"); self.wfile.flush()
            else:
                data = r.read()
                self.send_header("Content-Length", str(len(data))); self.end_headers()
                self.wfile.write(data); sent = len(data)
        except (BrokenPipeError, ConnectionResetError):
            log("%s %s model=%s -> %s %d: client went away after %d bytes" % (self.command, path, model or "-", name, r.status, sent))
            self.close_connection = True
        finally:
            c.close()
        log("%s %s model=%s -> %s %d %s%d bytes in %.1f s" % (self.command, path, model or "-", name, r.status,
                                                               "streamed " if stream else "", sent, time.time() - t0))

# ---- Claude Code's side: the settings that use the router ---------------------------------
def settings_for(port, model, label, behaves_as="claude-sonnet-5"):
    return {"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:%d" % port},
            # A row is keyed "model" (this build's schema: { model, label?, description?, behavesAs? });
            # a row keyed "id", as the docs show, is ignored with "This row was ignored".
            # behavesAs: the Claude model whose capabilities the CLI assumes for it (effort levels,
            # context window); without it an unknown id "isn't described by this version's model
            # catalog" and the picker calls it unsupported.
            "modelPicker": {"options": [{"model": model, "label": label, "behavesAs": behaves_as,
                                         "description": "qwfnfer on this machine, through qwfn-router"}]}}

def settings_path():
    return os.path.join(os.environ.get("CLAUDE_CONFIG_DIR") or os.path.expanduser("~/.claude"), "settings.json")

def configure(port, model, label, behaves_as, remove=False):
    p = settings_path()
    s = {}
    if os.path.exists(p):
        s = json.load(open(p))
        bak = p + ".bak-" + time.strftime("%Y%m%d-%H%M%S"); shutil.copy2(p, bak); print("backup:", bak)
    want = settings_for(port, model, label, behaves_as)
    env = s.setdefault("env", {})
    picker = s.get("modelPicker")
    opts = picker.get("options", []) if isinstance(picker, dict) else (picker if isinstance(picker, list) else [])
    opts = [o for o in opts if not (isinstance(o, dict) and model in (o.get("model"), o.get("id")))]
    if remove:
        env.pop("ANTHROPIC_BASE_URL", None)
        if not env: s.pop("env", None)
        if opts: s["modelPicker"] = dict(picker if isinstance(picker, dict) else {}, options=opts)
        else: s.pop("modelPicker", None)
    else:
        env["ANTHROPIC_BASE_URL"] = want["env"]["ANTHROPIC_BASE_URL"]
        s["modelPicker"] = dict(picker if isinstance(picker, dict) else {}, options=opts + want["modelPicker"]["options"])
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f: json.dump(s, f, indent=2); f.write("\n")
    print(("removed from" if remove else "written to") + " " + p + " (new sessions pick it up; running ones keep their base URL)")

def unit_path():
    return os.path.expanduser("~/.config/systemd/user/qwfn-router.service")

def install_service(args):
    here = os.path.abspath(__file__)
    unit = "\n".join([
        "[Unit]", "Description=qwfn-router: one Claude Code base URL for Anthropic and the local model", "After=network.target", "",
        "[Service]",
        # Quoted: systemd splits ExecStart on spaces, and a checkout path may hold some.
        'ExecStart="%s" "%s" --port %d --server "%s" --upstream "%s"' % (sys.executable, here, args.port, args.server, args.upstream),
        "Restart=always", "RestartSec=2", "",
        "[Install]", "WantedBy=default.target", ""])
    os.makedirs(os.path.dirname(unit_path()), exist_ok=True)
    open(unit_path(), "w").write(unit)
    for cmd in (["daemon-reload"], ["enable", "--now", "qwfn-router.service"]):
        r = subprocess.run(["systemctl", "--user"] + cmd, capture_output=True, text=True)
        if r.returncode: print("systemctl --user %s: %s" % (" ".join(cmd), (r.stderr or r.stdout).strip())); sys.exit(1)
    print("installed and started %s (systemctl --user status qwfn-router)" % unit_path())

def uninstall_service():
    subprocess.run(["systemctl", "--user", "disable", "--now", "qwfn-router.service"], capture_output=True)
    try: os.remove(unit_path()); print("removed", unit_path())
    except FileNotFoundError: print("no service installed")
    subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True)

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--port", type=int, default=8089); ap.add_argument("--server", default="http://127.0.0.1:8080")
    ap.add_argument("--upstream", default="https://api.anthropic.com")
    ap.add_argument("--local-model", action="append", default=[], help="a model id routed to the server (repeatable)")
    ap.add_argument("--picker-model", default="qwen3.8-flash-next", help="the id the /model picker entry selects")
    ap.add_argument("--picker-label", default="Qwen3.8-Flash-Next (local)")
    ap.add_argument("--behaves-as", default="claude-sonnet-5", help="the Claude model whose capabilities Claude Code assumes for the picker entry")
    ap.add_argument("--print-settings", action="store_true"); ap.add_argument("--configure", action="store_true")
    ap.add_argument("--unconfigure", action="store_true")
    ap.add_argument("--install-service", action="store_true"); ap.add_argument("--uninstall-service", action="store_true")
    a = ap.parse_args()
    if a.print_settings: print(json.dumps(settings_for(a.port, a.picker_model, a.picker_label, a.behaves_as), indent=2)); return
    if a.configure: configure(a.port, a.picker_model, a.picker_label, a.behaves_as); return
    if a.unconfigure: configure(a.port, a.picker_model, a.picker_label, a.behaves_as, remove=True); return
    if a.install_service: install_service(a); return
    if a.uninstall_service: uninstall_service(); return
    Handler.router = Router(a.server, a.upstream, a.local_model + [a.picker_model])
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    srv.daemon_threads = True
    log("listening on http://127.0.0.1:%d: %s for the local model, %s for the rest" % (a.port, a.server, a.upstream))
    try: srv.serve_forever()
    except KeyboardInterrupt: pass

if __name__ == "__main__":
    main()
