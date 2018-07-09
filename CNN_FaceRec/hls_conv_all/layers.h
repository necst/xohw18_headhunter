#pragma once

#include <cfloat>
#include "hls_stream.h"
#include "hls_video.h"
#include "NecstStream.hpp"

#define LOG_1(n) (((n) >= 2) ? 1 : 0)
#define LOG_2(n) (((n) >= 1<<2) ? (2 + LOG_1((n)>>2)) : LOG_1(n))
#define LOG_4(n) (((n) >= 1<<4) ? (4 + LOG_2((n)>>4)) : LOG_2(n))
#define LOG_8(n) (((n) >= 1<<8) ? (8 + LOG_4((n)>>8)) : LOG_4(n))
#define LOG_32(n)   (((n) >= 1<<16) ? (16 + LOG_8((n)>>16)) : LOG_8(n))

/* 
* Shift the window up, flushing the first row of the image.
* @param window: window to be shifted
* @param index: control flag to verify boundaries
*/
template <
    int FM_IN,      //number of input feature maps
    int KH,         //kernel height
    int DIMH,       //image height
    int DIMW,       //image width
    typename data_t //data type of the pixels
>
void convShiftData(hls::stream<data_t> &streamIn, hls::LineBuffer<KH, FM_IN*DIMW, data_t> &lineBuf){
	line_shift: for (int i = 0; i < FM_IN*DIMW; i++){
		#pragma HLS PIPELINE II=1
		data_t val = streamIn.read();

		lineBuf.shift_up(i);
		lineBuf.insert_top(val, i);
	}
}


/*
* Implement a convolutional layer of a CNN. Computation is performed
* on portions of the image at a time through a window.
* Each filter in the synthesized core is computed in parallel on all the
* input pixels.
* Layer dimensions should not be too big if there are limited resources.
* @param streamIn: input stream to load data
* @param streamOut: output stream to send results
* @param weights: buffer containing the weights of the layer
*/
template <
    int FM_IN,          //number of input feature maps
    int FM_OUT,         //number of output feature maps
    int KH,             //kernel height
    int KW,             //kernel width
	int KS,				//kernel stride
    int DIMH,           //image height
    int DIMW,           //image width
    typename data_in,   //data type of input pixels
    typename data_out   //data type of results, based on number of operations
>
void convLayer (hls::stream<data_in> &streamIn, hls::stream<data_out> &streamOut, data_in weights[FM_OUT][FM_IN][KH][KW], data_in bias[FM_OUT], hls::stream<my_ap_axis<ap_uint<32>,32,1,1,1> > &axiStreamOut){

    static hls::Window<KH, FM_IN*KW, data_in> window;
    static hls::LineBuffer<KH, FM_IN*DIMW, data_in> lineBuf;
    static hls::Window<FM_IN, KH+1, data_in> temp_window;

    //static hls::Window<FM_IN + LOG_32(FM_IN), 1, data_in> tmp_acc;
    data_in tmp_acc[4];

    data_out sum;

    union UNFI {
    	float a;
    	unsigned int b;
    } t ;

    my_ap_axis<ap_uint<32>,32,1,1,1> d;
    memsetAXIS_Data <my_ap_axis<ap_uint<32>,32,1,1,1> > (d,32);

    line_load: for(int i = 0; i < FM_IN*KH*DIMW; i++){
		#pragma HLS PIPELINE II=1
    	data_in val = streamIn.read();

    	lineBuf.shift_up(i % (DIMW*FM_IN) );
    	lineBuf.insert_top(val, i % (DIMW*FM_IN) );
    }

    fm_height: for (int i = 0; i < DIMH-KH+1; i+=KS) {
        fm_width: for (int j = 0; j < DIMW-KW+1; j+=KS) {

        	std::cout << "Calculate out pixel ( " << i/KS << " , " << j/KS << " ) \n";

            //Load window from line buffer
            std::cout << "Loading new window in registers to calculate out pixel ( " << i/KS << " , " << j/KS << " ) \n";
            window_load: for(int w = 0; w < FM_IN*KH*KW; w++) {
                int kh = w / (FM_IN*KW);
                int kw = w % (FM_IN*KW);
                window.insert(lineBuf.getval( kh, kw + j*FM_IN), kh, kw);
            }

            fm_out: for (int k = 0; k < FM_OUT; k++) {
                #pragma HLS LOOP_FLATTEN off
                sum = 0;
                fm_in: for (int l = 0; l < FM_IN; l++) {
                	#pragma HLS PIPELINE II=1
                    temp_window.insert(0, l, KH);
                    ker_height: for (int s = 0; s < KH; s++) {
                    	#pragma HLS UNROLL
                        temp_window.insert(0, l, s);
                        ker_width: for (int t = 0; t < KW; t++) {
							#pragma HLS UNROLL
                            data_out w = weights[k][l][s][t];
                            data_out pixel = window.getval(s, (t*FM_IN) + l);

                            data_out tmp_mul = w * pixel;

                            temp_window.insert(temp_window.getval(l, s) + tmp_mul, l, s);
                        }
                        temp_window.insert(temp_window.getval(l, s) + temp_window.getval(l, KH) ,l, KH);
                    }
                    
                    data_in val = temp_window.getval(l, KH);
    				if (l<4) {
    					tmp_acc[l] = val;//.insert(val, l, 0);
    				} else {
    					tmp_acc[l%4] += val; //insert(tmp_acc.getval(l%8,0) + val, l%8, 0);
    				}

    				//sum += temp_window.getval(l, KH);
                }
                
                sum = tmp_acc[0]+tmp_acc[1]+tmp_acc[2]+tmp_acc[3];

                /*** Reduction tree on tmp_acc ***/
/*
                tmp_acc_init: for (int l = 0; l < FM_IN-1; l+=2) {
					#pragma HLS PIPELINE II=1
                	tmp_acc.insert(temp_window.getval(l, KH) + temp_window.getval(l+1, KH), l/2, 0);
                }
                int last = FM_IN/2;

                if (FM_IN%2 == 1){
                	tmp_acc.insert(temp_window.getval(FM_IN-1, KH), FM_IN/2, 0);
                	last++;
                }

                int first = 0;
                int dx;
                reduce_tree: for (int it=0; it < LOG_32((FM_IN+1)/2)+1; it++) { //while(last - first > 1) {
					#pragma HLS PIPELINE II=1
                	//std::cout << "Reduce from " << first << " to " << last << " span = " << last-first << "\n";
                	reduce_step: for (int l = first; l < last-1; l+=2) {
						#pragma HLS UNROLL
                		tmp_acc.insert(tmp_acc.getval(l,0) + tmp_acc.getval(l+1,0), last+((l-first)/2), 0);
                	}

                	dx = last-first;
                	first = last;
                	if (dx%2 == 1){
                		tmp_acc.insert(tmp_acc.getval(last-1, 0), first + dx/2, 0);
                		last = last + (dx/2)+1;
                	} else {
                		last = last + dx/2;
                	}

                }

                sum = tmp_acc.getval(first,0);
*/
                /*** End reduction tree ***/

                /* ReLu */
                t.a = sum + bias[k] < 0 ? 0 : sum + bias[k];
                /* End ReLu*/

                if (i+KS >= DIMH-KH+1 && j+KS >= DIMW-KW+1 && k==FM_OUT-1)
                	d.last = 1;

                d.data = t.b;
                axiStreamOut << d;

            }
        }

        if (i+KS-1 < DIMH-KH) {
        	new_line_load:for(int ks = 0; ks < KS; ks++) {
        		convShiftData<FM_IN, KH, DIMH, DIMW, data_in> (streamIn, lineBuf);
        	}
        }
    }
}


