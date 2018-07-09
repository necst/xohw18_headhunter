#include <hls_stream.h>
#include "layers.h"
#include "io_utils.h"
#include "NecstStream.hpp"

#include "sizes.h"

template<
    unsigned int DATA_WIDTH_OUT,
    unsigned int DATA_WIDTH_IN,
    unsigned int NUM_ELEMENTS,
    typename data_t
>
void computation(hls::stream<data_t> &streamDat, hls::stream<my_ap_axis<ap_uint<DATA_WIDTH_OUT>,DATA_WIDTH_OUT,1,1,1> > &axiStream){
#pragma HLS DATAFLOW
    maxPoolLayer<FM_0, KH_1, KW_1, KS_1, DIMH_0, DIMW_0, SPAN, data_t> (streamDat, streamDat, axiStream);
}

void max1(hls::stream<my_ap_axis<ap_uint<DATA_WIDTH>,DATA_WIDTH,1,1,1> > &axiStreamIn,
        hls::stream<my_ap_axis<ap_uint<DATA_WIDTH>,DATA_WIDTH,1,1,1> > &axiStreamOut){

#pragma HLS INTERFACE axis register port=axiStreamIn
#pragma HLS INTERFACE axis register port=axiStreamOut
#pragma HLS INTERFACE ap_ctrl_none port=return

#pragma HLS DATAFLOW

	hls::stream<data_t> streamIn("In");
	hls::stream<data_t> streamOut("Out");
#pragma HLS STREAM variable=streamIn depth=7500 
#pragma HLS STREAM variable=streamOut depth=29 
	
	axis2Stream<MEM_WIDTH, DATA_WIDTH, 1, data_t> (axiStreamIn, streamIn, ELEMENTS_IN);
	
	computation<RES_WIDTH, DATA_WIDTH, 1, data_t> (streamIn, axiStreamOut);
}

