#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
out=runs/feedback_vision_20261023/board_deploy
test ! -e "$out"
test -s runs/feedback_vision_20261023/board_build/reflex_vision.bin
test -s /lib/firmware/xilinx/multiple_doublevadd_publisher/shell.json
# Neither the new overlay nor this script contains an actuator port or CAN call.
# Refuse while userspace has the old accelerator open.
if fuser /dev/dri/renderD128 /dev/zocl 2>/dev/null; then
  echo 'An accelerator consumer is active; no reconfiguration performed' >&2
  exit 1
fi
mkdir -p "$out" build/pl_vision
xmutil listapps > "$out/apps_before.txt"
ps -eo pid,ppid,stat,args > "$out/processes_before.txt"
gcc -O2 -shared -fPIC reflex/pl_vision_driver.c -o build/pl_vision/libplvision.so
dtc -@ -I dts -O dtb -o "$out/reflex-vision.dtbo" fpga/reflex_vision.dts
install -d /lib/firmware/xilinx/reflex-vision
install -m 644 runs/feedback_vision_20261023/board_build/reflex_vision.bin /lib/firmware/xilinx/reflex-vision/reflex-vision.bit.bin
install -m 644 "$out/reflex-vision.dtbo" /lib/firmware/xilinx/reflex-vision/reflex-vision.dtbo
printf '{"shell_type":"XRT_FLAT","num_slots":"1"}\n' > /lib/firmware/xilinx/reflex-vision/shell.json
sha256sum /lib/firmware/xilinx/reflex-vision/* > "$out/firmware_hashes.txt"
xmutil unloadapp
if ! xmutil loadapp reflex-vision; then
  xmutil loadapp multiple_doublevadd_publisher
  exit 1
fi
xmutil listapps > "$out/apps_after.txt"
cat /sys/class/fpga_manager/fpga0/state > "$out/fpga_state.txt"
nohup python3 scripts/pl_vision_service.py --out runs/feedback_vision_20261023/board_service --seconds 7200 > "$out/service.log" 2>&1 < /dev/null &
echo $! > "$out/service.pid"
