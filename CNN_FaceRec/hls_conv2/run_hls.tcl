open_project -reset Conv2
set_top conv2
add_files cnn.cpp
add_files io_utils.h
add_files layers.h
add_files NecstStream.hpp
add_files sizes.h
add_files -tb cnn_cpu.h
add_files -tb test_bench.cpp
open_solution "solution1"
set_part {xc7z020clg400-1} -tool vivado
create_clock -period 10 -name default
csim_design 
csynth_design
#cosim_design -trace_level all 
export_design -rtl verilog -format ip_catalog
exit
