#include <iostream>
#include <ap_int.h>
#include "io_utils.h"
#include "cnn_cpu.h"

#include "sizes.h"

using namespace std;

void max2(hls::stream<my_ap_axis<ap_uint<DATA_WIDTH>,DATA_WIDTH,1,1,1> > &axiStreamIn,
        hls::stream<my_ap_axis<ap_uint<DATA_WIDTH>,DATA_WIDTH,1,1,1> > &axiStreamOut);

int main()
{
	data_t img[SAMPLES][ELEMENTS_IN];

	hls::stream<data_t> streamOut;
	hls::stream<data_t> streamIn;

	hls::stream<my_ap_axis<ap_uint<DATA_WIDTH>,DATA_WIDTH,1,1,1> > axiStreamOut;
	hls::stream<my_ap_axis<ap_uint<DATA_WIDTH>,DATA_WIDTH,1,1,1> > axiStreamIn;

	data_t res_fpga[SAMPLES][ELEMENTS_OUT];
	data_t res_cpu_temp[SAMPLES][ELEMENTS_OUT];
	data_t res_cpu[SAMPLES][ELEMENTS_OUT];

	int count = 0, count_cpu = 0, count_fpga = 0;

	srand(0);
	for(int i = 0; i < SAMPLES; i++){
		for(int p = 0; p < ELEMENTS_IN; p++){
			img[i][p] = (data_t) (( (float) rand() / ((float) (RAND_MAX)) ));
		}
	}

	cout << "Done!" << endl << flush;

	//Execution
	for(int s = 0; s < SAMPLES; s++){
		for (int i = 0; i < DIMH_0; i++) {
			for (int j = 0; j < DIMW_0; j++) {
				for (int l = 0; l < FM_0; l++) {
					int idx = l * DIMH_0 * DIMW_0 + i * DIMW_0 + j;
					streamIn << img[s][idx];
				}
			}
		}
	}

	stream2Axis<DATA_WIDTH, DATA_WIDTH, ELEMENTS_IN, data_t> (streamIn, axiStreamIn, SAMPLES);

	cout << "FPGA exec\n" << flush;

	max2(axiStreamIn, axiStreamOut);

	my_ap_axis<ap_uint<DATA_WIDTH>, DATA_WIDTH,1,1,1> packet ;
	float* res;
	for(int s = 0; s < SAMPLES; s++){
		for (int i = 0; i < DIMH_1; i++) {
			for (int j = 0; j < DIMW_1; j++) {
				for (int l = 0; l < FM_1; l++) {
					int idx = l * DIMH_1 * DIMW_1 + i * DIMW_1 + j;
					packet = streamPopAxi<my_ap_axis< ap_uint<DATA_WIDTH>,DATA_WIDTH,1,1,1 >,hls::stream<my_ap_axis< ap_uint<DATA_WIDTH>,DATA_WIDTH,1,1,1> > > (axiStreamOut);

					res = reinterpret_cast<data_t*>( &(packet.data) );
					res_fpga[s][idx] = *res;
				}
			}
		}

		// CPU Execution

		pool_layer(DIM_0, K_1, FM_0, KS_1, img[s], res_cpu_temp[s]);
		localResponseNorm<FM_1, DIMH_1, DIMW_1>(res_cpu_temp[s], res_cpu[s]);

	}

	for(int i=0;i<SAMPLES;i++){
		for (int j=0;j<FM_1*DIMW_1*DIMH_1;j++) {
			if (abs(res_cpu[i][j] - res_fpga[i][j]) > 1e-3 ) {
				printf("Index : %d Res CPU : %7.7f != Res FPGA : %7.7f \n", j, res_cpu[i][j], res_fpga[i][j]);
				count++;
			} else if (FM_1*DIMW_1*DIMH_1 < 100) {
			    printf("Index : %d Res CPU : %7.7f == Res FPGA : %7.7f \n", j, res_cpu[i][j], res_fpga[i][j]);
			}
		}
	}
	
	printf("%d mispredictions (out of %u samples )\n", count, FM_1*DIMW_1*DIMH_1);

	if(count){
		cout << "Test failed!" << endl;
		return 1;
	}

	cout << "Test passed!" << endl;

	return 0;
}
