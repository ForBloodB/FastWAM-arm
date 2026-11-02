#!/bin/bash
set -eu
cd /home/ubuntu/reflex-arm
name="$1"
[[ "$name" =~ ^[a-z0-9_]+$ ]]
amplitude="$2"
duration="$3"
repetitions="$4"
trap 'systemctl start reflex-motorbridge-gateway.service' EXIT
systemctl stop reflex-motorbridge-gateway.service
PYTHONPATH=. python3 scripts/wrist_csp_bench.py --joint 6 --out "runs/j6_repair_20261020/$name" \
  --j6-speed-kp 2.5 --wrist-amplitude "$amplitude" --wrist-duration "$duration" \
  --wrist-peak-hold 20 --wrist-return-hold 8 --announce-peak --repetitions "$repetitions" \
  --execute-authorized-test
