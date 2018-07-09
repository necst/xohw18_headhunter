#pragma once

#include <ap_int.h>
#include <hls_stream.h>
#include "NecstStream.hpp"

template<
    unsigned int AXI_DATA_WIDTH,
    unsigned int WORDS
>
void AxiMaster2Stream(ap_uint<AXI_DATA_WIDTH> *memoryPort, hls::stream<ap_uint<AXI_DATA_WIDTH> > &out, unsigned int numImages){
    for(unsigned int i=0; i<numImages; i++){
    #pragma HLS LOOP_FLATTEN off
        for(int w = 0; w < WORDS; w++){
    #pragma HLS PIPELINE II=1
            ap_uint<AXI_DATA_WIDTH> data = memoryPort[w];
            out.write(data);
        }
    }
}

template<
    unsigned int AXI_DATA_WIDTH,
    unsigned int WORDS
>
void Stream2AxiMaster(hls::stream<ap_uint<AXI_DATA_WIDTH> > &in, ap_uint<AXI_DATA_WIDTH> *memoryPort, unsigned int numImages){
    for(unsigned int i=0; i<numImages; i++){
    #pragma HLS LOOP_FLATTEN off
        for(int w = 0; w < WORDS; w++){
#pragma HLS PIPELINE II=1
            ap_uint<AXI_DATA_WIDTH> data = in.read();
            memoryPort[w] = data;
        }
    }
}

template<
    unsigned int AXI_DATA_WIDTH,
    unsigned int WORDS
>
void AxiStream2Stream(hls::stream<my_ap_axis<ap_uint<AXI_DATA_WIDTH>,AXI_DATA_WIDTH,1,1,1> > &stream_in, hls::stream<ap_uint<AXI_DATA_WIDTH> > &stream_out, unsigned int numImages){
    for(int img=0; img<numImages; img++){
        for (int i = 0; i < WORDS; i++) {
    #pragma HLS PIPELINE II=1
            my_ap_axis<ap_uint<AXI_DATA_WIDTH>, AXI_DATA_WIDTH,1,1,1> packet = streamPopAxi<my_ap_axis<ap_uint<AXI_DATA_WIDTH>,
                                                                                    AXI_DATA_WIDTH,1,1,1>,
                                                                                    hls::stream<my_ap_axis<ap_uint<AXI_DATA_WIDTH>,
                                                                                    AXI_DATA_WIDTH,1,1,1> > >(stream_in);
            stream_out.write(packet.data);
        }
    }
}

template<
    unsigned int AXI_DATA_WIDTH,
    unsigned int WORDS
>
void Stream2AxiStream(hls::stream<ap_uint<AXI_DATA_WIDTH> > &stream_in, hls::stream<my_ap_axis<ap_uint<AXI_DATA_WIDTH>, AXI_DATA_WIDTH,1,1,1> > &stream_out, unsigned int numImages){
    for (int i = 0; i < WORDS * numImages; i++) {
#pragma HLS PIPELINE II=1
        ap_uint<AXI_DATA_WIDTH> data = stream_in.read();
        streamPush<ap_uint<AXI_DATA_WIDTH>,
            my_ap_axis<ap_uint<AXI_DATA_WIDTH>,
            AXI_DATA_WIDTH,1,1,1>,
            hls::stream<my_ap_axis<ap_uint<AXI_DATA_WIDTH>,
            AXI_DATA_WIDTH,1,1,1> > >(data, i == (WORDS*numImages - 1) ? 1 : 0,
            stream_out, AXI_DATA_WIDTH);
    }
}

/*
Cast a data type to another.
It works when in and out are of the same data width.
E.g.:
    - half <--> ap_uint<16>
    - float <--> ap_uint<32>
*/
template<
    unsigned int WIDTH
>
void valueCast(ap_uint<WIDTH> *in, ap_uint<WIDTH> *out){
#pragma HLS INLINE
        * ((ap_uint<WIDTH> *) out) =  * ((ap_uint<WIDTH> *) in);
}

/*
Cast a stream of data type to another.
It works when InType and OutType are of the same data width.
E.g.:
    - half <--> ap_uint<16>
    - float <--> ap_uint<32>
*/
template<
    unsigned int WORDS,
    unsigned int WIDTH,
    typename data_in,
    typename data_out
>
void streamCast(hls::stream<data_in> &in, hls::stream<data_out> &out, unsigned int numImages){
	for(int img=0; img<numImages; img++){
		for(int w = 0; w < WORDS; w++){
		#pragma HLS PIPELINE II=1
			data_in data = in.read();
			data_out tmp;
			valueCast<WIDTH>((ap_uint<WIDTH> *) &data, (ap_uint<WIDTH> *) &tmp);
			out.write(tmp);
		}
	}
}


template<
    unsigned int WORDS,
    unsigned int DATA_WIDTH_IN,
    unsigned int DATA_WIDTH_OUT
>
void dataPack(hls::stream<ap_uint<DATA_WIDTH_IN> > &in, hls::stream<ap_uint<DATA_WIDTH_OUT> > &out, unsigned int numImages){
    const unsigned int ELEMS = DATA_WIDTH_OUT / DATA_WIDTH_IN;
    const unsigned int PACKETS = WORDS / ELEMS;

    ap_uint<DATA_WIDTH_OUT> dataOut = 0;

    for(int img=0; img<numImages; img++){
		for(int p = 0; p < PACKETS; p++){
			for(int e = 0; e < ELEMS; e++){
		#pragma HLS PIPELINE II=1

				unsigned int lowBit = e * DATA_WIDTH_IN;
				unsigned int highBit = (e+1) * DATA_WIDTH_IN - 1;

				ap_uint<DATA_WIDTH_IN> dataIn = in.read();
				dataOut(highBit, lowBit) = dataIn;

				if(e == ELEMS-1){
					out.write(dataOut);
					dataOut = 0;
				}
			}
		}
    }
}

template<
    unsigned int WORDS,
    unsigned int DATA_WIDTH_IN,
    unsigned int DATA_WIDTH_OUT
>
void dataUnpack(hls::stream<ap_uint<DATA_WIDTH_IN> > &in, hls::stream<ap_uint<DATA_WIDTH_OUT> > &out, unsigned int numImages){
    unsigned int ELEMS = DATA_WIDTH_IN / DATA_WIDTH_OUT;

    ap_uint<DATA_WIDTH_IN> dataIn = 0;
    for(int img=0; img<numImages; img++){
		for(int p = 0; p < WORDS; p++){
			for(int e = 0; e < ELEMS; e++){
		#pragma HLS PIPELINE II=1

				if(e == 0)
					dataIn = in.read();

				ap_uint<DATA_WIDTH_OUT> dataOut = dataIn(DATA_WIDTH_OUT - 1, 0);
				out.write(dataOut);

				dataIn = dataIn >> DATA_WIDTH_OUT;
			}
		}
    }
}

template<
    unsigned int WORDS,
    unsigned int DATA_WIDTH_IN,
    unsigned int DATA_WIDTH_OUT
>
void dataWidthConverter(hls::stream<ap_uint<DATA_WIDTH_IN> > &in, hls::stream<ap_uint<DATA_WIDTH_OUT> > &out, unsigned int numImages){

    if(DATA_WIDTH_IN > DATA_WIDTH_OUT){
        dataUnpack<WORDS, DATA_WIDTH_IN, DATA_WIDTH_OUT>(in, out, numImages);
    } else {
        dataPack<WORDS, DATA_WIDTH_IN, DATA_WIDTH_OUT>(in, out, numImages);
    }
}

template<
    unsigned int MEM_WIDTH_IN,
    unsigned int DATA_WIDTH_OUT,
    unsigned int NUM_ELEMENTS,
    typename data_t
>
void mem2Stream(ap_uint<MEM_WIDTH_IN> *memIn, hls::stream<data_t> &streamIn, unsigned int numImages){
    hls::stream<ap_uint<MEM_WIDTH_IN> > memStream;
    hls::stream<ap_uint<DATA_WIDTH_OUT> > resizedStream;
    
    AxiMaster2Stream<MEM_WIDTH_IN, NUM_ELEMENTS/(MEM_WIDTH_IN/DATA_WIDTH_OUT)>
        (memIn, memStream, numImages);

    dataUnpack<NUM_ELEMENTS/(MEM_WIDTH_IN/DATA_WIDTH_OUT), MEM_WIDTH_IN, DATA_WIDTH_OUT>
        (memStream, resizedStream, numImages);

    streamCast<NUM_ELEMENTS, DATA_WIDTH_OUT, ap_uint<DATA_WIDTH_OUT>, data_t>
        (resizedStream, streamIn, numImages);
}

template<
    unsigned int MEM_WIDTH_OUT,
    unsigned int DATA_WIDTH_IN,
    unsigned int NUM_ELEMENTS,
    typename data_t
>
void stream2Mem(hls::stream<data_t> &streamOut, ap_uint<MEM_WIDTH_OUT> *memOut, unsigned int numImages){
    hls::stream<ap_uint<MEM_WIDTH_OUT> > memStream;
    hls::stream<ap_uint<DATA_WIDTH_IN> > resizedStream;

    streamCast<NUM_ELEMENTS, DATA_WIDTH_IN, data_t, ap_uint<DATA_WIDTH_IN> >
        (streamOut, resizedStream, numImages);

    dataPack<NUM_ELEMENTS, DATA_WIDTH_IN, MEM_WIDTH_OUT>
        (resizedStream, memStream, numImages);

    Stream2AxiMaster<MEM_WIDTH_OUT, NUM_ELEMENTS/(MEM_WIDTH_OUT/DATA_WIDTH_IN)>
        (memStream, memOut, numImages);
}

template<
    unsigned int DATA_WIDTH_IN,
    unsigned int DATA_WIDTH_OUT,
    unsigned int NUM_ELEMENTS,
    typename data_t
>
void axis2Stream(hls::stream<my_ap_axis<ap_uint<DATA_WIDTH_IN>,DATA_WIDTH_IN,1,1,1> > &axiStream, hls::stream<data_t> &streamIn, unsigned int numImages){
#pragma HLS DATAFLOW
	hls::stream<ap_uint<DATA_WIDTH_IN> > simpleStream;
    hls::stream<ap_uint<DATA_WIDTH_OUT> > resizedStream;
#pragma HLS STREAM variable=simpleStream depth=255 dim=1
#pragma HLS STREAM variable=resizedStream depth=255 dim=1

    AxiStream2Stream<DATA_WIDTH_IN, NUM_ELEMENTS/(DATA_WIDTH_IN/DATA_WIDTH_OUT)>
        (axiStream, simpleStream, numImages);

    dataUnpack<NUM_ELEMENTS/(DATA_WIDTH_IN/DATA_WIDTH_OUT), DATA_WIDTH_IN, DATA_WIDTH_OUT>
        (simpleStream, resizedStream, numImages);

    streamCast<NUM_ELEMENTS, DATA_WIDTH_OUT, ap_uint<DATA_WIDTH_OUT>, data_t>
        (resizedStream, streamIn, numImages);
}


template<
    unsigned int DATA_WIDTH_OUT,
    unsigned int DATA_WIDTH_IN,
    unsigned int NUM_ELEMENTS,
    typename data_t
>
void stream2Axis(hls::stream<data_t> &streamOut, hls::stream<my_ap_axis<ap_uint<DATA_WIDTH_OUT>,DATA_WIDTH_OUT,1,1,1> > &axiStream, unsigned int numImages){
#pragma HLS DATAFLOW
	hls::stream<ap_uint<DATA_WIDTH_OUT> > simpleStream;
    hls::stream<ap_uint<DATA_WIDTH_IN> > resizedStream;
#pragma HLS STREAM variable=simpleStream depth=255 dim=1
#pragma HLS STREAM variable=resizedStream depth=255 dim=1

    streamCast<NUM_ELEMENTS, DATA_WIDTH_IN, data_t, ap_uint<DATA_WIDTH_IN> >
        (streamOut, simpleStream, numImages);

    dataPack<NUM_ELEMENTS, DATA_WIDTH_IN, DATA_WIDTH_OUT>
        (simpleStream, resizedStream, numImages);

    Stream2AxiStream<DATA_WIDTH_OUT, NUM_ELEMENTS/(DATA_WIDTH_OUT/DATA_WIDTH_IN)>
        (resizedStream, axiStream, numImages);
}

