#!/usr/bin/env bash
# Board-only diagnostic app load; no actuator or CAN command is sent.
set -eu
cd "$(dirname "$0")/.."
test -s runs/diagnostic_build/reflex_diagnostic.bin
test -f fpga/reflex_diagnostic.dts
test -f /lib/firmware/xilinx/k26-starter-kits/k26_starter_kits.bit.bin
mkdir -p runs/pl_board
dtc -@ -I dts -O dtb -o runs/pl_board/reflex-diagnostic.dtbo fpga/reflex_diagnostic.dts
install -d /lib/firmware/xilinx/reflex-diagnostic
install -m 644 runs/diagnostic_build/reflex_diagnostic.bin /lib/firmware/xilinx/reflex-diagnostic/reflex-diagnostic.bit.bin
install -m 644 runs/pl_board/reflex-diagnostic.dtbo /lib/firmware/xilinx/reflex-diagnostic/reflex-diagnostic.dtbo
printf '{"shell_type":"XRT_FLAT","num_slots":"1"}\n' > /lib/firmware/xilinx/reflex-diagnostic/shell.json
xmutil listapps > runs/pl_board/apps_before.txt
xmutil unloadapp
if ! xmutil loadapp reflex-diagnostic; then
    xmutil loadapp k26-starter-kits
    exit 1
fi
sleep 1
xmutil listapps > runs/pl_board/apps_after.txt
cat /sys/class/fpga_manager/fpga0/state
for p in /sys/class/uio/uio*/name; do printf '%s ' "$p"; cat "$p"; done
python3 scripts/test_pl_board.py
