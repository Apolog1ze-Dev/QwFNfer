#!/usr/bin/env bash
# The qwfn console: pick a downloaded model, get settings for this machine, start/stop the server, watch it live.
cd "$(dirname "$0")/.." && exec python3 tools/qwfn_console.py "$@"
