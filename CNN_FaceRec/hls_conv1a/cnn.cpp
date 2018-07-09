#include <hls_stream.h>
#include "layers.h"
#include "io_utils.h"
#include "NecstStream.hpp"

#include "sizes.h"

static data_t W_conv[FM_1][FM_0][KH_1][KW_1];
static data_t b_conv[FM_1];

union UNFI {
    float a;
    unsigned int b;
};


template<
    unsigned int DATA_WIDTH_OUT,
    unsigned int DATA_WIDTH_IN,
    unsigned int NUM_ELEMENTS,
    typename data_t
>
void computation(hls::stream<data_t> &streamDat, hls::stream<my_ap_axis<ap_uint<DATA_WIDTH_OUT>,DATA_WIDTH_OUT,1,1,1> > &axiStream){
#pragma HLS DATAFLOW
    convLayer<FM_0, FM_1, KH_1, KW_1, KS_1, DIMH_0, DIMW_0, data_t, data_t> (streamDat, streamDat, W_conv, b_conv, axiStream);
}


void load_Wconv(hls::stream<data_t> &streamIn) {
	for(int fm_out = 0; fm_out < FM_1; fm_out++){
		for(int fm_in = 0; fm_in < FM_0; fm_in++){
			for(int kh = 0; kh < KH_1; kh++){
				for(int kw = 0; kw < KW_1; kw++){
					W_conv[fm_out][fm_in][kh][kw] = streamIn.read();
				}
			}
		}
	}

	for(int fm_out = 0; fm_out < FM_1; fm_out++){
		b_conv[fm_out] = streamIn.read();
	}
}


void split(hls::stream<data_t> &streamIn, hls::stream<data_t> &streamW, hls::stream<data_t> &streamDat, int datElem, int wElem) {
	#pragma HLS PIPELINE

	for (int i=0;i<wElem;i++) {
		streamW << streamIn.read();
	}

	for (int i=0;i<datElem;i++) {
		streamDat << streamIn.read();
	}
}


void conv1a(hls::stream<my_ap_axis<ap_uint<DATA_WIDTH>,DATA_WIDTH,1,1,1> > &axiStreamIn,
        hls::stream<my_ap_axis<ap_uint<DATA_WIDTH>,DATA_WIDTH,1,1,1> > &axiStreamOut, int load){

#pragma HLS INTERFACE axis register port=axiStreamIn
#pragma HLS INTERFACE axis register port=axiStreamOut
#pragma HLS INTERFACE s_axilite register port=return bundle=control
#pragma HLS INTERFACE s_axilite register port=load bundle=control

#pragma HLS ARRAY_PARTITION variable=W_conv complete dim=4

#pragma HLS DATAFLOW

	hls::stream<data_t> streamIn("In");
	hls::stream<data_t> streamOut("Out");
#pragma HLS STREAM variable=streamIn depth=7500 
#pragma HLS STREAM variable=streamOut depth=29 

	hls::stream<data_t> streamW("WIn");
	hls::stream<data_t> streamDat("DatIn");
#pragma HLS STREAM variable=streamW depth=60
#pragma HLS STREAM variable=streamDat depth=60
	
	axis2Stream<MEM_WIDTH, DATA_WIDTH, 1, data_t> (axiStreamIn, streamIn, load!=COMPUTE ? (FM_1*FM_0*KH_1*KW_1)+FM_1 : ELEMENTS_IN );

	split(streamIn,streamW,streamDat, load==COMPUTE ? ELEMENTS_IN : 0, load==COMPUTE ? 0 : (FM_1*FM_0*KH_1*KW_1)+FM_1 );

	if (load!=COMPUTE) {
		load_Wconv(streamW);
	}
	else {
		computation<RES_WIDTH, DATA_WIDTH, 1, data_t> (streamDat, axiStreamOut);
	}
}

