#!/usr/bin/env bash
# No CAN commands or external PL actuator ports. Old diagnostic app retained.
set -eu
cd "$(dirname "$0")/.."
test -s runs/framework_20261020/board_build/reflex_board_v3.bin
test -s /lib/firmware/xilinx/reflex-diagnostic/reflex-diagnostic.bit.bin
test -f fpga/reflex_board_v3.dts
test ! -e runs/framework_20261020/board_v3
# Refuse overlay changes while any of our motion tests is running.
if pgrep -f '[s]cripts/wrist_csp_bench.py' >/dev/null; then
    echo 'Motion test active; refusing FPGA reconfiguration' >&2
    exit 1
fi
mkdir -p runs/framework_20261020/board_deploy
xmutil listapps > runs/framework_20261020/board_deploy/apps_before.txt
dtc -@ -I dts -O dtb -o runs/framework_20261020/board_deploy/reflex-board-v3.dtbo fpga/reflex_board_v3.dts
install -d /lib/firmware/xilinx/reflex-board-v3
install -m 644 runs/framework_20261020/board_build/reflex_board_v3.bin /lib/firmware/xilinx/reflex-board-v3/reflex-board-v3.bit.bin
install -m 644 runs/framework_20261020/board_deploy/reflex-board-v3.dtbo /lib/firmware/xilinx/reflex-board-v3/reflex-board-v3.dtbo
printf '{"shell_type":"XRT_FLAT","num_slots":"1"}\n' > /lib/firmware/xilinx/reflex-board-v3/shell.json
sha256sum /lib/firmware/xilinx/reflex-board-v3/* > runs/framework_20261020/board_deploy/firmware_hashes.txt
xmutil unloadapp
if ! xmutil loadapp reflex-board-v3; then
    xmutil loadapp reflex-diagnostic
    exit 1
fi
sleep 1
xmutil listapps > runs/framework_20261020/board_deploy/apps_after.txt
if ! python3 scripts/test_pl_board_v3.py --out runs/framework_20261020/board_v3; then
    xmutil unloadapp
    xmutil loadapp reflex-diagnostic
    exit 1
fi
cat /sys/class/fpga_manager/fpga0/state
