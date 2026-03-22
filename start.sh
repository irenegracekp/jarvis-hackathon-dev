#!/bin/bash
# Start the full Jarvis/EVA pipeline: daemon + tunnel + main
# Usage: bash start.sh

set -e
cd "$(dirname "$0")"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info() { echo -e "${GREEN}[start]${NC} $*"; }
warn() { echo -e "${YELLOW}[start]${NC} $*"; }

# Cleanup on exit
cleanup() {
    info "Shutting down..."
    [ -n "$DAEMON_PID" ] && kill $DAEMON_PID 2>/dev/null
    [ -n "$TUNNEL_PID" ] && kill $TUNNEL_PID 2>/dev/null
    [ -n "$MAIN_PID" ] && kill $MAIN_PID 2>/dev/null
    wait 2>/dev/null
    info "Done."
}
trap cleanup EXIT INT TERM

# Activate venv
source venv/bin/activate

# 1) Start Reachy daemon in background
info "Starting Reachy daemon..."
reachy-mini-daemon --headless --deactivate-audio > /tmp/reachy-daemon.log 2>&1 &
DAEMON_PID=$!
sleep 2

if kill -0 $DAEMON_PID 2>/dev/null; then
    info "Reachy daemon running (pid $DAEMON_PID)"
else
    warn "Reachy daemon failed to start. Check /tmp/reachy-daemon.log"
    warn "Continuing without robot..."
    DAEMON_PID=""
fi

# 2) Start cloudflare tunnel in background and capture URL
info "Starting cloudflare tunnel on port 8001..."
cloudflared tunnel --url http://localhost:8001 > /tmp/tunnel.log 2>&1 &
TUNNEL_PID=$!

# Wait for tunnel URL
TUNNEL_URL=""
for i in $(seq 1 15); do
    TUNNEL_URL=$(grep -oP 'https://[a-z0-9-]+\.trycloudflare\.com' /tmp/tunnel.log 2>/dev/null | head -1)
    if [ -n "$TUNNEL_URL" ]; then
        break
    fi
    sleep 1
done

if [ -n "$TUNNEL_URL" ]; then
    info "Tunnel ready: $TUNNEL_URL"
    # Update .env with new tunnel URL
    sed -i "s|^MCP_PUBLIC_URL=.*|MCP_PUBLIC_URL=$TUNNEL_URL|" .env
    info "Updated .env with tunnel URL"
else
    warn "Tunnel failed to start. OpenClaw commands won't work."
    warn "Check /tmp/tunnel.log"
    sed -i "s|^MCP_PUBLIC_URL=.*|MCP_PUBLIC_URL=|" .env
fi

# 3) Start main pipeline
info "Starting Jarvis pipeline..."
echo ""
python3 main.py --agora --no-vlm
MAIN_PID=$!
