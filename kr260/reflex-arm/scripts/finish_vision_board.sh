#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
out=runs/feedback_vision_20261023
touch "$out/load_stop"
pid=$(cat "$out/board_deploy/service.pid")
if kill -0 "$pid" 2>/dev/null; then
  grep -zq 'scripts/pl_vision_service.py' "/proc/$pid/cmdline"
  kill -TERM "$pid"
  for _ in 1 2 3 4 5; do
    if ! kill -0 "$pid" 2>/dev/null; then break; fi
    sleep .2
  done
fi
if ss -ltn | grep -q ':18766 '; then echo 'Image service still listening' >&2; exit 1; fi
xmutil listapps > "$out/board_deploy/apps_before_restore.txt"
xmutil unloadapp
xmutil loadapp multiple_doublevadd_publisher
xmutil listapps > "$out/board_deploy/apps_restored.txt"
ufw delete allow proto tcp from 172.21.16.96 to 172.21.16.34 port 18766
ufw status numbered > "$out/board_deploy/firewall_restored.txt"
ps -eo pid,args | grep '[w]s_gateway' > "$out/board_deploy/motor_service_unchanged.txt"
python3 - <<'PY'
from pathlib import Path
import json,time
p=Path('runs/feedback_vision_20261023/board_deploy')
(p/'restoration.json').write_text(json.dumps({'restored_application':'multiple_doublevadd_publisher','temporary_firewall_rule_removed':True,'image_service_stopped':True,'motor_commands':0,'wall_unix':time.time()},indent=2))
PY
