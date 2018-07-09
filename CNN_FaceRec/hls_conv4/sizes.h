#pragma once

#define SAMPLES 1

#define FM_0 384
#define DIM_0 13
#define DIMW_0 DIM_0
#define DIMH_0 DIM_0

#define FM_1 24  //384 -> 16*24
#define K_1 3
#define KW_1 K_1
#define KH_1 K_1
#define KS_1 1
#define KP_1 1

#define DIM_1 ((((DIM_0 + 2*KP_1) - K_1)/KS_1) + 1) //13
#define DIMW_1 DIM_1
#define DIMH_1 DIM_1

typedef float data_t;

#define DATA_WIDTH 32
#define ELEMENTS_IN (FM_0*DIMW_0*DIMH_0)
#define ELEMENTS_OUT (FM_1*DIMW_1*DIMH_1)
#define MEM_WIDTH DATA_WIDTH
#define RES_WIDTH DATA_WIDTH

#define COMPUTE 0
#define CONV_WEIGHT 1
