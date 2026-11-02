# Out-of-context synthesis only. Does not create or install a board bitstream.
# Vivado: vivado -mode batch -source fpga/synthesize.tcl
set root [file normalize [file join [file dirname [info script]] ..]]
set_param general.maxThreads 4
foreach top {safety_fsm linear_interpolator} {
    file mkdir $root/runs/synthesis/$top
    create_project -in_memory $top -part xck26-sfvc784-2LV-c
    read_verilog -sv $root/fpga/safety_fsm.sv
    read_verilog -sv $root/fpga/linear_interpolator.sv
    synth_design -top $top -mode out_of_context -part xck26-sfvc784-2LV-c
    create_clock -name control_clk -period 10.000 [get_ports clk]
    report_utilization -file $root/runs/synthesis/$top/utilization.rpt
    report_timing_summary -file $root/runs/synthesis/$top/timing.rpt
    write_checkpoint -force $root/runs/synthesis/$top/$top.dcp
    close_project
}
puts "OOC synthesis complete; no placement, routing, pin constraints or board timing verified"
