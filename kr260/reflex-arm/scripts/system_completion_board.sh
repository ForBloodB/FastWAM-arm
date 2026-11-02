#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/reflex-arm
mode=${1:?cpu/mmio/dma/restore};tag=${2:-$mode}
out=runs/system_completion_20261029/board
mkdir -p "$out"
python3 - "$out" <<'PY'
import sys,subprocess,json,os,signal,time
from pathlib import Path
r=Path(sys.argv[1]);p=r/'service.pid'
if p.exists():
 pid=int(p.read_text());cmd=Path('/proc')/str(pid)/'cmdline'
 if cmd.exists():
  raw=cmd.read_bytes()
  if b'scripts/task_image_service.py' not in raw or b'system_completion_20261029' not in raw:raise RuntimeError('PID ownership mismatch')
  os.kill(pid,signal.SIGTERM)
  for _ in range(30):
   if not cmd.exists() or not cmd.read_bytes():break
   time.sleep(.1)
  else:raise RuntimeError('Task image service failed to stop')
if not (r/'original.json').exists():
 text=subprocess.check_output(['xmutil','listapps'],text=True)
 active=[line.split()[0] for line in text.splitlines() if line.split() and line.split()[-1].rstrip(',')=='0']
 assert len(active)==1,active
 ports=subprocess.check_output(['ss','-ltn'],text=True);assert ':18768 ' not in ports
 gateways=subprocess.run(['pgrep','-f','ws_gateway'],capture_output=True,text=True).stdout.split()
 states={p:Path('/proc',p,'stat').read_text() for p in gateways if Path('/proc',p,'stat').exists()}
 (r/'original.json').write_text(json.dumps(dict(active_app=active[0],gateways=states,apps=text,firewall=subprocess.check_output(['ufw','status'],text=True)),indent=2))
print(json.loads((r/'original.json').read_text())['active_app'])
PY
if ss -ltn | grep -q ':18768 ';then echo 'Reserved task port busy' >&2;exit 1;fi
original=$(python3 -c "import json;print(json.load(open('$out/original.json'))['active_app'])")
if test "$mode" != cpu;then
 if fuser /dev/uio* /dev/reflex-image-dma /dev/zocl 2>/dev/null;then echo 'Accelerator consumer active' >&2;exit 1;fi
 if lsmod | grep -q '^reflex_image_dma ';then rmmod reflex_image_dma;fi
 case "$mode" in mmio) app=reflex-vision;;dma) app=reflex-dma;;restore) app=$original;;*) exit 2;;esac
 xmutil unloadapp
 if ! xmutil loadapp "$app";then xmutil loadapp "$original";exit 1;fi
 if test "$mode" = dma;then insmod fpga/dma_driver/reflex_image_dma.ko;test -c /dev/reflex-image-dma;fi
fi
xmutil listapps > "$out/apps_$tag.txt"
if test "$mode" = restore;then
 if test -f "$out/firewall_added";then ufw delete allow proto tcp from 172.21.16.96 to 172.21.16.34 port 18768;fi
 python3 - "$out" <<'PY'
import json,sys
from pathlib import Path
r=Path(sys.argv[1]);v=json.loads((r/'original.json').read_text());checks={p:Path('/proc',p,'stat').exists() for p in v['gateways']}
(r/'restored.json').write_text(json.dumps(dict(original_app=v['active_app'],gateways_still_present=checks,motor_commands=0,temporary_service_stopped=True,temporary_port_rule_removed=True),indent=2))
assert all(checks.values())
PY
 exit 0
fi
if ufw status | grep -q '^Status: active';then
 ufw allow proto tcp from 172.21.16.96 to 172.21.16.34 port 18768
 touch "$out/firewall_added"
fi
nohup taskset -c 3 python3 scripts/task_image_service.py --backend "$mode" --out "$out/service_$tag" --port 18768 --seconds 7200 > "$out/service_$tag.log" 2>&1 < /dev/null &
echo $! > "$out/service.pid"
sleep .5
cat "$out/service_$tag.log"
