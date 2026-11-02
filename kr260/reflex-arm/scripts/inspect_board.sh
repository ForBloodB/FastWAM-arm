#!/usr/bin/env bash
set -u
hostname; uname -a
printf '\nFPGA STATE\n'
cat /sys/class/fpga_manager/fpga0/state 2>/dev/null
printf '\nUIO NAMES\n'
for p in /sys/class/uio/uio*/name; do printf '%s ' "$p"; cat "$p"; done
printf '\nFIRMWARE\n'
find /lib/firmware/xilinx -maxdepth 2 -type f 2>/dev/null | head -30
printf '\nUSB\n'
lsusb
printf '\nINTERFACES\n'
ip -br link
printf '\nVERSIONS\n'
python3 --version
printf '\nDEVICE MODEL\n'
tr '\0' '\n' < /proc/device-tree/model
