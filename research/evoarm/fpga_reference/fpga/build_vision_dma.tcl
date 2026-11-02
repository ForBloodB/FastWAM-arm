set root [file normalize [file join [file dirname [info script]] ..]]
set out $root/runs/residual_dma_20261023/board_build
file mkdir $out
set_param general.maxThreads 4
create_project reflex_dma $out/project -part xck26-sfvc784-2LV-c
set_property board_part xilinx.com:kr260_som:part0:1.1 [current_project]
add_files $root/fpga/vision_axis.sv
set_property file_type Verilog [get_files $root/fpga/vision_axis.sv]
create_bd_design reflex_dma
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:* ps]
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1"} $ps
set_property -dict [list CONFIG.PSU__USE__M_AXI_GP0 {1} CONFIG.PSU__USE__M_AXI_GP1 {0} CONFIG.PSU__USE__M_AXI_GP2 {0} CONFIG.PSU__USE__S_AXI_GP2 {1} CONFIG.PSU__USE__IRQ0 {1} CONFIG.PSU__FPGA_PL0_ENABLE {1} CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {100}] $ps
set reset [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:* reset]
connect_bd_net [get_bd_pins ps/pl_clk0] [get_bd_pins reset/slowest_sync_clk] [get_bd_pins ps/maxihpm0_fpd_aclk] [get_bd_pins ps/saxihp0_fpd_aclk]
connect_bd_net [get_bd_pins ps/pl_resetn0] [get_bd_pins reset/ext_reset_in]
set one [create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:* one]
set_property CONFIG.CONST_VAL 1 $one
connect_bd_net [get_bd_pins one/dout] [get_bd_pins reset/dcm_locked]
set core [create_bd_cell -type module -reference vision_axis core]
connect_bd_net [get_bd_pins ps/pl_clk0] [get_bd_pins core/aclk]
connect_bd_net [get_bd_pins reset/peripheral_aresetn] [get_bd_pins core/aresetn]
set dma [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:* dma]
set_property -dict [list CONFIG.c_include_sg {0} CONFIG.c_sg_length_width {23} CONFIG.c_addr_width {32} CONFIG.c_m_axi_mm2s_data_width {64} CONFIG.c_m_axi_s2mm_data_width {64} CONFIG.c_m_axis_mm2s_tdata_width {32} CONFIG.c_s_axis_s2mm_tdata_width {32} CONFIG.c_mm2s_burst_size {64} CONFIG.c_s2mm_burst_size {64} CONFIG.c_include_mm2s_dre {1} CONFIG.c_include_s2mm_dre {1}] $dma
connect_bd_intf_net [get_bd_intf_pins dma/M_AXIS_MM2S] [get_bd_intf_pins core/S_AXIS]
connect_bd_intf_net [get_bd_intf_pins core/M_AXIS] [get_bd_intf_pins dma/S_AXIS_S2MM]
connect_bd_net [get_bd_pins ps/pl_clk0] [get_bd_pins dma/s_axi_lite_aclk] [get_bd_pins dma/m_axi_mm2s_aclk] [get_bd_pins dma/m_axi_s2mm_aclk]
connect_bd_net [get_bd_pins reset/peripheral_aresetn] [get_bd_pins dma/axi_resetn]
set ctrl [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:* ctrl]
set_property -dict [list CONFIG.NUM_SI 1 CONFIG.NUM_MI 2] $ctrl
set mem [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:* mem]
set_property -dict [list CONFIG.NUM_SI 2 CONFIG.NUM_MI 1] $mem
foreach x {ctrl mem} {
 connect_bd_net [get_bd_pins ps/pl_clk0] [get_bd_pins $x/aclk]
 connect_bd_net [get_bd_pins reset/interconnect_aresetn] [get_bd_pins $x/aresetn]
}
connect_bd_intf_net [get_bd_intf_pins ps/M_AXI_HPM0_FPD] [get_bd_intf_pins ctrl/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins ctrl/M00_AXI] [get_bd_intf_pins dma/S_AXI_LITE]
connect_bd_intf_net [get_bd_intf_pins dma/M_AXI_MM2S] [get_bd_intf_pins mem/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins dma/M_AXI_S2MM] [get_bd_intf_pins mem/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins mem/M00_AXI] [get_bd_intf_pins ps/S_AXI_HP0_FPD]
set gpio [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_gpio:* identity]
set_property -dict [list CONFIG.C_GPIO_WIDTH 32 CONFIG.C_GPIO2_WIDTH 32 CONFIG.C_ALL_INPUTS 1 CONFIG.C_ALL_INPUTS_2 1 CONFIG.C_IS_DUAL 1] $gpio
connect_bd_net [get_bd_pins core/signature] [get_bd_pins identity/gpio_io_i]
connect_bd_net [get_bd_pins core/frames] [get_bd_pins identity/gpio2_io_i]
connect_bd_intf_net [get_bd_intf_pins ctrl/M01_AXI] [get_bd_intf_pins identity/S_AXI]
connect_bd_net [get_bd_pins ps/pl_clk0] [get_bd_pins identity/s_axi_aclk]
connect_bd_net [get_bd_pins reset/peripheral_aresetn] [get_bd_pins identity/s_axi_aresetn]
set irq [create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:* irq]
set_property CONFIG.NUM_PORTS 2 $irq
connect_bd_net [get_bd_pins dma/mm2s_introut] [get_bd_pins irq/In0]
connect_bd_net [get_bd_pins dma/s2mm_introut] [get_bd_pins irq/In1]
connect_bd_net [get_bd_pins irq/dout] [get_bd_pins ps/pl_ps_irq0]
assign_bd_address -offset 0xA0020000 -range 64K -target_address_space [get_bd_addr_spaces ps/Data] [get_bd_addr_segs dma/S_AXI_LITE/Reg] -force
assign_bd_address -offset 0xA0010000 -range 64K -target_address_space [get_bd_addr_spaces ps/Data] [get_bd_addr_segs identity/S_AXI/Reg] -force
assign_bd_address
validate_bd_design
save_bd_design
generate_target all [get_files reflex_dma.bd]
add_files -norecurse [make_wrapper -files [get_files reflex_dma.bd] -top]
set_property top reflex_dma_wrapper [current_fileset]
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {error "Implementation failed"}
open_run impl_1
report_utilization -file $out/utilization.rpt
report_timing_summary -report_unconstrained -file $out/timing.rpt
report_drc -file $out/drc.rpt
write_hw_platform -fixed -include_bit -force -file $out/reflex_dma.xsa
set bit [get_property DIRECTORY [get_runs impl_1]]/reflex_dma_wrapper.bit
file copy $bit $out/reflex_dma.bit
write_cfgmem -force -format bin -interface SMAPx32 -disablebitswap -loadbit "up 0 $bit" -file $out/reflex_dma.bin
puts "DMA_BUILD_SUCCESS $out"
