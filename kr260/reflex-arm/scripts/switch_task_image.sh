#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mode=${1:?cpu/mmio/dma/restore};tag=${2:-$mode}
out=runs/residual_dma_20261023
mkdir -p "$out/deploy"
if test -f "$out/deploy/service.pid"; then
 pid=$(cat "$out/deploy/service.pid")
 if test -r "/proc/$pid/cmdline" && grep -zq 'scripts/task_image_service.py' "/proc/$pid/cmdline"; then
  kill -TERM "$pid"
  for x in 1 2 3 4 5 6 7 8 9 10; do
   if ! test -r "/proc/$pid/cmdline" || ! grep -zq 'scripts/task_image_service.py' "/proc/$pid/cmdline"; then break; fi
   sleep .3
  done
 fi
fi
if ss -ltn | grep -q ':18767 '; then echo 'Image service still active' >&2;exit 1;fi
if test ! -f "$out/deploy/apps_original.txt"; then
 xmutil listapps > "$out/deploy/apps_original.txt"
 ps -eo pid,args > "$out/deploy/processes_original.txt"
 ufw status numbered > "$out/deploy/firewall_original.txt"
fi
if test "$mode" != cpu; then
 if fuser /dev/dri/renderD128 /dev/zocl /dev/reflex-image-dma 2>/dev/null; then echo 'Active accelerator consumer' >&2;exit 1;fi
 if lsmod | grep -q '^reflex_image_dma ';then rmmod reflex_image_dma;fi
 case "$mode" in
  mmio) app=reflex-vision;;
  dma) app=reflex-dma;;
  restore) app=multiple_doublevadd_publisher;;
  *) exit 2;;
 esac
 xmutil unloadapp
 if ! xmutil loadapp "$app";then xmutil loadapp multiple_doublevadd_publisher;exit 1;fi
 if test "$mode" = dma;then insmod fpga/dma_driver/reflex_image_dma.ko;test -c /dev/reflex-image-dma;fi
 xmutil listapps > "$out/deploy/apps_$tag.txt"
fi
if test "$mode" = restore;then
 ufw delete allow proto tcp from 172.21.16.96 to 172.21.16.34 port 18767
 ufw status numbered > "$out/deploy/firewall_restored.txt"
 ps -eo pid,args | grep '[w]s_gateway' > "$out/deploy/motor_service_after.txt"
 echo '{"restored":true,"motor_commands":0,"temporary_port_removed":true}' > "$out/deploy/restoration.json"
 exit 0
fi
ufw allow proto tcp from 172.21.16.96 to 172.21.16.34 port 18767
nohup taskset -c 3 python3 scripts/task_image_service.py --backend "$mode" --out "$out/service_$tag" --seconds 7200 > "$out/deploy/service_$tag.log" 2>&1 < /dev/null &
echo $! > "$out/deploy/service.pid"
sleep .4
cat "$out/deploy/service_$tag.log"
