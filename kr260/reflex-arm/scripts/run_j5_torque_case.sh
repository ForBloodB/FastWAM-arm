#!/bin/bash
set -eu
cd /home/ubuntu/reflex-arm
name="$1"
[[ "$name" =~ ^[a-z0-9_]+$ ]]
repetitions="$2"
current="${3:-0.6}"
trap 'systemctl start reflex-motorbridge-gateway.service' EXIT
systemctl stop reflex-motorbridge-gateway.service
PYTHONPATH=. python3 scripts/wrist_csp_bench.py --joint 5 --out "runs/j6_repair_20261020/$name" \
  --j5-torque-limit 0.6 --j5-current "$current" --wrist-amplitude 2 --wrist-duration 40 \
  --wrist-peak-hold 20 --wrist-return-hold 8 --announce-peak --repetitions "$repetitions" \
  --execute-authorized-test
