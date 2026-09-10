#!/usr/bin/env bash
# Relays the LiteLLM gateway's Bastion tunnel onto an interface Docker can
# reach, for docker-compose users (see docker-compose.yml's `extra_hosts` on
# the api/worker services, and CLAUDE.md's "Running it" section).
#
# Why this exists: the tunnel opened by the Foundry landing-zone toolkit
# (`az network bastion tunnel`, wrapped by `deploy.sh gateway`/`foundry`)
# always binds to 127.0.0.1 on the host -- there is no flag to bind it wider.
# Docker's `extra_hosts: ["<gateway-host>:host-gateway"]` resolves the
# gateway hostname to the host's real IP as seen from the container, but a
# listener bound strictly to loopback will refuse that connection -- this
# reproduces on Docker Desktop (Mac/Windows/WSL2) as well as native Linux
# Docker, not just one platform. The fix is a plain TCP relay: listen on all
# interfaces, forward to the loopback-only tunnel. socat does this as a
# transparent byte-for-byte passthrough, so TLS (hostname/cert checks the
# app makes to the real gateway hostname) is unaffected.
#
# Usage:
#   ./scripts/gateway-relay.sh <tunnel-port> [relay-port]
#
# <tunnel-port>  the local port your Bastion tunnel is already listening on
#                (127.0.0.1:<tunnel-port>) -- e.g. the port `foundry`/
#                `deploy.sh gateway` printed when it connected.
# [relay-port]   port to expose on all interfaces; defaults to <tunnel-port>.
#                Only pass a different value if that port is already taken.
#
# Leave this running in its own terminal for as long as `docker compose up`
# needs the gateway. Ctrl-C to stop it.

set -euo pipefail

if [[ $# -lt 1 ]]; then
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
