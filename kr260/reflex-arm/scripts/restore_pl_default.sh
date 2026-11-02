#!/usr/bin/env bash
set -eu
test -f /lib/firmware/xilinx/k26-starter-kits/k26_starter_kits.bit.bin
xmutil unloadapp
xmutil loadapp k26-starter-kits
xmutil listapps
