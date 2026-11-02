#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
mkdir -p runs/bridge
if [ -f runs/bridge/pid ] && kill -0 "$(cat runs/bridge/pid)" 2>/dev/null; then
  echo 'Bridge already running'
  exit 0
fi
nohup python3 -m reflex.bridge --bind 172.21.16.34 --allow-peer 172.21.16.96 \
  --log runs/bridge/packets.jsonl > runs/bridge/stdout.log 2>&1 < /dev/null &
echo "$!" > runs/bridge/pid
echo "Diagnostic bridge PID $! (NO ACTUATORS)"
