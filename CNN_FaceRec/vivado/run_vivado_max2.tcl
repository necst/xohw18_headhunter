create_project -force Max2 ./Max2 -part xc7z020clg400-1
set_property board_part www.digilentinc.com:pynq-z1:part0:1.0 [current_project]
set_property ip_repo_paths ./../ [current_project]
update_ip_catalog
create_bd_design "design_1"
update_compile_order -fileset sources_1
startgroup
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0
endgroup
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 -config {make_external "FIXED_IO, DDR" apply_board_preset "1" Master "Disable" Slave "Disable" }  [get_bd_cells processing_system7_0]
startgroup
set_property -dict [list CONFIG.PCW_USE_S_AXI_HP0 {1}] [get_bd_cells processing_system7_0]
endgroup
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0
set_property -dict [list CONFIG.c_include_sg {0} CONFIG.c_sg_include_stscntrl_strm {0}] [get_bd_cells axi_dma_0]
set_property -dict [list CONFIG.c_sg_length_width {23}] [get_bd_cells axi_dma_0]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config {Master "/axi_dma_0/M_AXI_MM2S" Clk "Auto" }  [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config {Master "/processing_system7_0/M_AXI_GP0" Clk "Auto" }  [get_bd_intf_pins axi_dma_0/S_AXI_LITE]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config {Slave "/processing_system7_0/S_AXI_HP0" Clk "Auto" }  [get_bd_intf_pins axi_dma_0/M_AXI_S2MM]
startgroup
create_bd_cell -type ip -vlnv xilinx.com:hls:max2:1.0 max_2
endgroup
connect_bd_intf_net [get_bd_intf_pins max_2/axiStreamIn] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
connect_bd_intf_net [get_bd_intf_pins max_2/axiStreamOut] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
connect_bd_net [get_bd_pins max_2/ap_clk] [get_bd_pins processing_system7_0/FCLK_CLK0]
connect_bd_net [get_bd_pins max_2/ap_rst_n] [get_bd_pins rst_ps7_0_100M/peripheral_aresetn]
#apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config {Master "/processing_system7_0/M_AXI_GP0" intc_ip "/ps7_0_axi_periph" Clk "Auto" }  [get_bd_intf_pins max_2/s_axi_control]
validate_bd_design
make_wrapper -files [get_files ./Max2/Max2.srcs/sources_1/bd/design_1/design_1.bd] -top
add_files -norecurse ./Max2/Max2.srcs/sources_1/bd/design_1/hdl/design_1_wrapper.v
save_bd_design

launch_runs synth_1 -jobs 20
launch_runs impl_1 -jobs 20
launch_runs impl_1 -to_step write_bitstream -jobs 20
wait_on_run impl_1
file copy -force ./Max2/Max2.runs/impl_1/design_1_wrapper.bit ./max_2.bit
write_bd_tcl -force max_2.tcl
exit

