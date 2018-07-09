#pragma once

#include <cfloat>
#include "hls_stream.h"
#include "hls_video.h"
#include "NecstStream.hpp"
#include <hls_math.h>

#define LOG_1(n) (((n) >= 2) ? 1 : 0)
#define LOG_2(n) (((n) >= 1<<2) ? (2 + LOG_1((n)>>2)) : LOG_1(n))
#define LOG_4(n) (((n) >= 1<<4) ? (4 + LOG_2((n)>>4)) : LOG_2(n))
#define LOG_8(n) (((n) >= 1<<8) ? (8 + LOG_4((n)>>8)) : LOG_4(n))
#define LOG_32(n)   (((n) >= 1<<16) ? (16 + LOG_8((n)>>16)) : LOG_8(n))

/*
 * Shift @param window up, flushing the first row of the image.
 * @param streamIn: input stream to load new data
 * @param window: window to be shifted
 * @param index: control flag to verify boundaries
 */
template <
    int FM_IN,      //number of input feature maps
    int PH,         //kernel height
    int DIMH,       //image height
    int DIMW,       //image width
    int PS,         //kernel shift step
    typename data_t //data type of the pixels
>
void poolShiftData(hls::stream<data_t> &streamIn, hls::LineBuffer<PH, FM_IN*DIMW, data_t> &lineBuf){
	line_shift: for (int i = 0; i < FM_IN*DIMW; i++){
		#pragma HLS PIPELINE II=1
		data_t val = streamIn.read();

		lineBuf.shift_up(i);
		lineBuf.insert_top(val, i);
	}
}

/*
* Implement a max-pooling layer of a CNN. Computation is performed
* on portions of the image at a time through a window.
* @param streamIn: input stream to load data
* @param streamOut: output stream to send results
*/
template<
    unsigned int FM_IN,
    unsigned int PH,
    unsigned int PW,
    unsigned int PS,
    unsigned int DIMH,
    unsigned int DIMW,
    typename data_t
>
void maxPoolLayer (hls::stream<data_t> &streamIn, hls::stream<data_t> &streamOut, hls::stream<my_ap_axis<ap_uint<32>,32,1,1,1> > &axiStreamOut){
    
	union UNFI {
		float a;
		unsigned int b;
	} t ;
	my_ap_axis<ap_uint<32>,32,1,1,1> d;
	memsetAXIS_Data <my_ap_axis<ap_uint<32>,32,1,1,1> > (d,32);
	float out;

	static hls::LineBuffer<PH, FM_IN*DIMW, data_t> lineBuf;
    //static hls::Window<PH, FM_IN*PW, data_t> window;
    static hls::Window<PW + LOG_32(PW), 1, data_t> tmp_max;

    static data_t max_arr[3];
#pragma HLS ARRAY_PARTITION variable=max_arr complete



    line_load: for(int i = 0; i < FM_IN*PH*DIMW; i++){
		#pragma HLS PIPELINE II=1
    	data_t val = streamIn.read();

    	lineBuf.shift_up(i % (DIMW*FM_IN) );
    	lineBuf.insert_top(val, i % (DIMW*FM_IN) );
    }

    fm_height: for (int i = 0; i < DIMH-PH+1; i+=PS) {
    	fm_width: for (int j = 0; j < DIMW-PW+1; j+=PS) {

    		std::cout << "Calculate out pixel ( " << i/PS << " , " << j/PS << " ) \n";

    		//Load window from line buffer
    		/*std::cout << "Loading new window in registers to calculate out pixel ( " << i/PS << " , " << j/PS << " ) \n";
    		window_load: for(int w = 0; w < FM_IN*PH*PW; w++) {
    			int ph = w / (FM_IN*PW);
    			int pw = w % (FM_IN*PW);
    			window.insert(lineBuf.getval( ph, pw + j*FM_IN), ph, pw);
    		}*/

    		fm_in: for (int fm = 0; fm < FM_IN; fm++) {
				#pragma HLS PIPELINE II=1
    			data_t max = -FLT_MAX;
    			ker_width: for (int pw = 0; pw < PW; pw++) {
					#pragma HLS UNROLL
    				tmp_max_init: for (int ph = 0; ph < PH-1; ph+=2) {
						#pragma HLS UNROLL
    					data_t pixel1 = lineBuf.getval(ph, ((j+pw)*FM_IN) + fm ); //window.getval(ph, pw*FM_IN + fm);
    					data_t pixel2 = lineBuf.getval(ph+1, ((j+pw)*FM_IN) + fm ); //window.getval(ph, (pw+1)*FM_IN + fm);
    					tmp_max.insert(pixel1>pixel2 ? pixel1 : pixel2, ph/2, 0);
    				}
    				int last = PH/2;

    				if (PH%2 == 1){
    					tmp_max.insert(lineBuf.getval(PH-1, (j+pw)*FM_IN + fm ), PH/2, 0); //window.getval(ph, (PW-1)*FM_IN + fm)
    					last++;
    				}

    				int first = 0;
    				int dx;
    				reduce_tree: for (int it=0; it < LOG_32((PW+1)/2)+1; it++) { //while(last - first > 1) {
						#pragma HLS UNROLL
    					//std::cout << "Reduce from " << first << " to " << last << " span = " << last-first << "\n";
    					reduce_step: for (int pw = first; pw < last-1; pw+=2) {
							#pragma HLS UNROLL
    						data_t pixel1 = tmp_max.getval(pw, 0);
    						data_t pixel2 = tmp_max.getval(pw+1, 0);
    						tmp_max.insert(pixel1>pixel2 ? pixel1 : pixel2, last+((pw-first)/2), 0);
    					}

    					dx = last-first;
    					first = last;
    					if (dx%2 == 1){
    						tmp_max.insert(tmp_max.getval(last-1, 0), first + dx/2, 0);
    						last = last + (dx/2)+1;
    					} else {
    						last = last + dx/2;
    					}

    				}

    				max_arr[pw] = tmp_max.getval(first, 0) ;
    			}

    			max = std::max(std::max(max_arr[0], max_arr[1]), max_arr[2]);

    			t.a = max;
    			d.data = t.b;

    			if (i+PS >= DIMH-PH+1 && j+PS >= DIMW-PW+1 && fm == FM_IN-1)
    				d.last = 1;

    			axiStreamOut << d;

    		}
    	}

    	if (i+PS-1 < DIMH-PH) // because there isn't padding in this layer
    		new_line_load:for(int ps = 0; ps < PS; ps++) {
    			poolShiftData<FM_IN, PH, DIMH, DIMW, PS, data_t> (streamIn, lineBuf);
    		}
    }
}


