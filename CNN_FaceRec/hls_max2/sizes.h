#pragma once

#define SAMPLES 1

#define FM_0 256
#define DIM_0 27
#define DIMW_0 DIM_0
#define DIMH_0 DIM_0

#define FM_1 FM_0
#define K_1 3
#define KW_1 K_1
#define KH_1 K_1
#define KS_1 2

#define DIM_1 (((DIM_0 - K_1)/KS_1) + 1) //13
#define DIMW_1 DIM_1
#define DIMH_1 DIM_1

#define SPAN 5
#define OFFSET 2
#define ALPHA 0.0001
#define BETA 0.75

typedef float data_t;

#define DATA_WIDTH 32
#define ELEMENTS_IN (FM_0*DIMW_0*DIMH_0)
#define ELEMENTS_OUT (FM_1*DIMW_1*DIMH_1)
#define MEM_WIDTH DATA_WIDTH
#define RES_WIDTH DATA_WIDTH

