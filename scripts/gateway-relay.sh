#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <tunnel-port> [relay-port]" >&2
  exit 1
fi

TUNNEL_PORT="$1"
RELAY_PORT="${2:-$TUNNEL_PORT}"

if ! command -v socat >/dev/null 2>&1; then
  echo "socat is required. Install it first:" >&2
  echo "  Debian/Ubuntu/WSL: sudo apt install socat" >&2
  echo "  macOS:             brew install socat" >&2
  exit 1
fi

echo "Relaying 0.0.0.0:${RELAY_PORT} -> 127.0.0.1:${TUNNEL_PORT} (Ctrl-C to stop)"
exec socat TCP-LISTEN:"${RELAY_PORT}",fork,reuseaddr TCP:127.0.0.1:"${TUNNEL_PORT}"
