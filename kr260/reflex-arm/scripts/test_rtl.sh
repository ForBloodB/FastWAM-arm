#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
mkdir -p runs/rtl
if command -v iverilog >/dev/null; then
  iv=(iverilog); vvp=(vvp)
else
  iv=("$PWD/tools/iverilog/usr/bin/iverilog" -B "$PWD/tools/iverilog/usr/lib/x86_64-linux-gnu/ivl")
  vvp=("$PWD/tools/iverilog/usr/bin/vvp" -M "$PWD/tools/iverilog/usr/lib/x86_64-linux-gnu/ivl")
fi
"${iv[@]}" -g2012 -s tb_safety -o runs/rtl/safety fpga/safety_fsm.sv fpga/tb_safety.sv
(cd runs/rtl; "${vvp[@]}" safety)
"${iv[@]}" -g2012 -s tb_interpolator -o runs/rtl/interpolator fpga/linear_interpolator.sv fpga/tb_interpolator.sv
"${vvp[@]}" runs/rtl/interpolator
"${iv[@]}" -g2012 -s tb_diagnostic -o runs/rtl/diagnostic fpga/safety_fsm.sv fpga/linear_interpolator.sv fpga/diagnostic_core.sv fpga/tb_diagnostic.sv
"${vvp[@]}" runs/rtl/diagnostic
