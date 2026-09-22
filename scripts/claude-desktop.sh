#!/usr/bin/env bash
# claude-desktop.sh: Claude Desktop on the local model.
#
# Claude Desktop has a third-party inference mode that takes any server speaking the
# Anthropic Messages API (this one does: POST /v1/messages). This script starts
# qwfn-server through the console if it is not running, writes a Claude Desktop profile
# whose inference provider is that server, and launches Claude Desktop on that profile.
# The profile is separate from your normal one (Electron's --user-data-dir), so the Claude
# Desktop signed into your claude.ai account is untouched and can keep running next to it.
# The app keeps third-party data in a "-3p" sibling of its user-data directory and moves
# there once a configuration is applied, so the profile is two directories:
# ~/.config/Claude-qwfnfer (the launch directory, holding the instance lock) and
# ~/.config/Claude-qwfnfer-3p (the configuration, chats and sessions).
#
#   scripts/claude-desktop.sh                 start the engine if needed, launch the app
#   scripts/claude-desktop.sh --restart       quit the local-model instance first (after a change)
#   scripts/claude-desktop.sh --stop          quit the local-model instance
#   scripts/claude-desktop.sh --no-server     do not start the engine; it must be running already
#   scripts/claude-desktop.sh --reset         delete the profile (its chats, sessions and settings)
#   --model ID          the model id sent to the server (default claude-sonnet-5: the app only
#                       accepts ids that look like Claude models; the server ignores the name,
#                       and the picker shows the model the server really serves)
#   --server-port N     the engine's port (8080)      --console-port N   the console's (8090)
#   --profile DIR       the profile directory (default ~/.config/Claude-qwfnfer)
#
# Works from the release bundle (this file under scripts/ next to bin/ and tools/) and
# from a source checkout.
set -u
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
[ -f "$HERE/tools/qwfn_console.py" ] || HERE=$(cd "$HERE/.." && pwd)
[ -f "$HERE/tools/qwfn_console.py" ] || { echo "claude-desktop.sh: tools/qwfn_console.py not found next to $0" >&2; exit 1; }

server_port=8080; console_port=8090; model="claude-sonnet-5"; profile="${XDG_CONFIG_HOME:-$HOME/.config}/Claude-qwfnfer"
start_server=1; restart=0; stop=0; reset=0
while [ $# -gt 0 ]; do
    case "$1" in
        --server-port)  server_port=$2; shift ;;
        --console-port) console_port=$2; shift ;;
        --model)        model=$2; shift ;;
        --profile)      profile=$2; shift ;;
        --no-server)    start_server=0 ;;
        --restart)      restart=1 ;;
        --stop)         stop=1 ;;
        --reset)        reset=1 ;;
        -h|--help)      sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "claude-desktop.sh: unknown option $1" >&2; exit 1 ;;
    esac
    shift
done
say() { printf 'claude-desktop: %s\n' "$*"; }
die() { printf 'claude-desktop: %s\n' "$*" >&2; exit 1; }
server="http://127.0.0.1:$server_port"
console="http://127.0.0.1:$console_port"
up() { curl -s -m 2 "$1/health" >/dev/null 2>&1; }

# ---- the running instance on this profile (Electron's single-instance lock) --------------
instance_pid() {
    local d t pid
    for d in "$profile" "$profile-3p"; do
        t=$(readlink "$d/SingletonLock" 2>/dev/null) || continue
        pid=${t##*-}
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then echo "$pid"; return 0; fi
    done
    return 1
}
quit_instance() {
    local pid; pid=$(instance_pid) || return 0
    say "quitting the local-model Claude Desktop (pid $pid)"
    kill -TERM "$pid" 2>/dev/null
    for _ in $(seq 1 50); do kill -0 "$pid" 2>/dev/null || return 0; sleep 0.2; done
    kill -KILL "$pid" 2>/dev/null; sleep 0.5
}
if [ $stop -eq 1 ]; then quit_instance; exit 0; fi
if [ $reset -eq 1 ]; then
    quit_instance
    [ -d "$profile" ] || [ -d "$profile-3p" ] || { say "no profile at $profile"; exit 0; }
    if [ -t 0 ]; then read -r -p "delete $profile and $profile-3p (its chats, sessions and settings)? [y/N] " a; [ "$a" = y ] || exit 1; fi
    rm -rf "$profile" "$profile-3p"; say "removed $profile and $profile-3p"; exit 0
fi
command -v claude-desktop >/dev/null 2>&1 || die "claude-desktop is not on PATH (this script drives the Linux Claude Desktop package)"

# ---- the engine ----------------------------------------------------------------------------
if ! up "$server"; then
    [ $start_server -eq 1 ] || die "no server at $server; start one from the console (or drop --no-server)"
    if ! curl -s -m 2 "$console/api/status" >/dev/null 2>&1; then
        say "starting the console at $console with the last served model"
        [ -f "$HERE/bin/libggml-base.so.0" ] && export LD_LIBRARY_PATH="$HERE/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        mkdir -p "$HOME/.cache/qwfn-console"
        (cd "$HERE" && setsid nohup python3 tools/qwfn_console.py --port "$console_port" --server-port "$server_port" --start \
            >"$HOME/.cache/qwfn-console/console.log" 2>&1 &)
        for _ in $(seq 1 30); do curl -s -m 2 "$console/api/status" >/dev/null 2>&1 && break; sleep 1; done
        curl -s -m 2 "$console/api/status" >/dev/null 2>&1 || die "the console did not come up; see ~/.cache/qwfn-console/console.log"
    else
        last=$(curl -s -m 5 "$console/api/config" | python3 -c 'import json,sys; l=json.load(sys.stdin).get("last") or {}; print(l.get("model",""), l.get("preset","coding"))')
        set -- $last
        [ -n "${1:-}" ] || die "the console has never served a model: open $console, pick one and start it once"
        say "starting the engine through the console: $(basename "$1") ($2)"
        r=$(curl -s -m 30 -X POST "$console/api/start" -H 'Content-Type: application/json' -d "{\"model\": \"$1\", \"preset\": \"$2\"}")
        python3 -c 'import json,sys; d=json.loads(sys.argv[1]); e=d.get("error"); sys.exit(1) if e and print("claude-desktop: " + e, file=sys.stderr) is None else 0' "$r" || exit 1
    fi
    say "waiting for the engine at $server (a cold start loads the model: up to a few minutes)"
    for _ in $(seq 1 240); do up "$server" && break; sleep 2; done
    up "$server" || die "the engine did not answer at $server; see ~/.cache/qwfn-console/server.log"
fi
served=$(curl -s -m 5 "$server/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null)
[ -n "$served" ] || served=qwen3.8-flash-next

# ---- the profile: a config library with one applied entry ---------------------------------
# The same files the in-app form (Developer -> Configure Third-Party Inference) writes.
lib="$profile-3p/configLibrary"
mkdir -p "$profile" "$lib"; chmod 700 "$profile" "$profile-3p" "$lib"
changed=$(python3 - "$lib" "$server" "$model" "$served" <<'PY'
import json, os, sys, uuid
lib, server, model, served = sys.argv[1:5]
meta_path = os.path.join(lib, "_meta.json")
meta = {"appliedId": "", "entries": []}
try: meta = json.load(open(meta_path))
except Exception: pass
ids = [e["id"] for e in meta.get("entries", []) if e.get("name") == "qwfnfer"]
cid = ids[0] if ids else str(uuid.uuid4())
conf = {
    "inferenceProvider": "gateway",
    "inferenceGatewayBaseUrl": server,
    "inferenceGatewayApiKey": "local",
    "inferenceGatewayAuthScheme": "bearer",
    "inferenceModels": [{"name": model, "labelOverride": served + " (local)", "anthropicFamilyTier": "sonnet", "isFamilyDefault": True}],
    "modelDiscoveryEnabled": False,
    # A silent stretch is a prefill: 17K tokens of Claude Code's first turn take ~30 s, a
    # 200K document minutes. The server pings every 15 s; this is how long the app waits past
    # its own five minutes of pings for the first real token. Range 300-1800.
    "inferenceStreamIdleTimeoutSec": 1800,
    "toolSearchEnabled": False,
}
path = os.path.join(lib, cid + ".json")
old = None
try: old = json.load(open(path))
except Exception: pass
if old != conf:
    with open(path, "w") as f: json.dump(conf, f, indent=2)
    os.chmod(path, 0o600)
if not ids: meta.setdefault("entries", []).append({"id": cid, "name": "qwfnfer"})
applied_before = meta.get("appliedId")
meta["appliedId"] = cid
with open(meta_path, "w") as f: json.dump(meta, f, indent=2)
os.chmod(meta_path, 0o600)
print("changed" if old != conf or applied_before != cid else "same")
PY
)

# ---- launch --------------------------------------------------------------------------------
if [ $restart -eq 1 ]; then quit_instance; fi
if pid=$(instance_pid); then
    [ "$changed" = changed ] && say "the profile changed; the running instance (pid $pid) keeps its old settings until you run with --restart"
    say "the local-model Claude Desktop is already running (pid $pid): bringing it to the front"
else
    say "launching Claude Desktop on $server ($served, sent as $model), profile $profile"
fi
setsid nohup claude-desktop --user-data-dir="$profile" >"$profile/launch.log" 2>&1 < /dev/null &
disown 2>/dev/null || true
say "log: $profile/launch.log"
