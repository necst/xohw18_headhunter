#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <cfloat>
#include <cmath>

int classify(float *data, int classes){
    int predicted;
    float max;
    float x, v;
    float tmp[classes];

    //Classification (LogSoftMax):
    max = -FLT_MAX;
    Class1:for (int i = 0; i < classes; i++) {
        if (data[i] > max) {
            max = data[i];
        }
    }

    x = 0;
    Class2:for (int i = 0; i < classes; i++) {
        v = data[i] - max;
        data[i] = exp(v);
        x += data[i];
    }

    Class3:for (int i = 0; i < classes; i++) {
        tmp[i] = data[i] / x;
        tmp[i] = log(tmp[i]);
    }

    //Prediction:
    max = -FLT_MAX;
    predicted = -1;
    Pred:for (int i = 0; i < classes; i++) {
        if (tmp[i] > max) {
            max = tmp[i];
            predicted = i;
        }
    }

    return predicted;
}


template<size_t in_ch, size_t out_ch, size_t kdim>
void conv_layer_cpu(int dim, int stride, float *in, float *out, float w[out_ch][in_ch][kdim][kdim], float *b, int padding) {

    int outdim = (dim-kdim+1);
    if(padding)
        outdim = dim;

    if(stride > 1) outdim = ((dim-kdim+1) / stride) + 1; //outdim = dim / stride;
    
    outdim = (((dim - kdim)/stride) + 1);

    for (int y = 0; y < outdim; y++) {
        for (int x = 0; x < outdim; x++) {
            for (int outfm = 0; outfm < (int)out_ch; outfm++) {
                int idx = outfm * outdim * outdim + y * outdim + x;
                float out_val = b[outfm];
                for(int infm = 0; infm < (int)in_ch; infm++) {
                    for (int ky = 0; ky < (int)kdim; ky++) {
                        for(int kx = 0; kx < (int)kdim; kx++) {
                            //padding
                            int yy = y * stride + ky;
                            int xx = x * stride + kx;
                            if(padding){
                                xx -= kdim/2;
                                yy -= kdim/2;
                            }
                            if(xx >= 0 && xx < dim && yy >= 0 &&yy < dim){
                                int in_idx = infm * dim * dim + yy * dim + xx;
                                out_val += in[in_idx] * w[outfm][infm][ky][kx];
                            }
                        }
                    }
                }
                out[idx] = out_val;
            }
        }
    }
}


void relu_layer(int dim, int channels, float *in, float *out) {
    int total_items = dim*dim*channels;

    for(int i = 0; i < total_items; i++) {
        out[i] = in[i] > 0 ? in[i] : 0;
    }
}


void pool_layer(int dim, int kdim, int channels, int stride, float *in, float *out) {

    int outdim = (dim-kdim+1);

    if(stride > 1) outdim = ((dim-kdim+1) / stride) + 1;

    for (int fm = 0; fm < channels; fm++) {
        for (int y = 0; y < outdim; y++) {
            for (int x = 0; x < outdim; x++) {
                float max_val = in[fm * dim * dim + y * stride * dim + x * stride];

                for (int ky = 0; ky < kdim; ky++) {
                    for(int kx = 0; kx < kdim; kx++) {
                        int in_idx = fm * dim * dim + (y * stride + ky) * dim + (x * stride + kx);
                        if (in[in_idx] > max_val) max_val = in[in_idx];
                    }
                }

                int idx = fm * outdim * outdim + y * outdim + x;
                out[idx] = max_val;
            }
        }
    }

}

template <int FM, int DIMH, int DIMW>
void localResponseNorm(float *fm, float *fm_norm)
{
    //Declaration and definition of the hyper-parameters n, k, alpha and beta
    int n=5;
    float k = 2;
    float alpha = 0.0001;
    float beta = 0.75;
    
    float half_n = n / 2;

    double sum;
    double unit_scale = 0;
    double scale = 0;
    int lower_i, upper_i;

    for(int i = 0; i < FM; i++){
        lower_i = ((0 > (i - half_n)) ? 0 : i - half_n);
        upper_i = (FM - 1 < i + half_n ? FM - 1 : i + half_n); 
        for(int x = 0; x < DIMH; x++){
            for(int y = 0; y < DIMW; y++){
                sum = 0;
                for(int j = lower_i; j <= upper_i ; j++){
                    sum += pow(fm[j*(DIMH*DIMW) + x*DIMW + y], 2);
                }
                unit_scale = k + alpha * sum;
                scale = pow(unit_scale, -beta);

                fm_norm[i*(DIMH*DIMW) + x*DIMW + y] = fm[i*(DIMH*DIMW) + x*DIMW + y] * scale;
            }
        }
    }
}


template<size_t in_dim, size_t out_dim>
void lin_layer(float *in, float *out, float w[out_dim][in_dim], float *b) {

    for (int out_idx = 0; out_idx < out_dim; out_idx++) {
        float out_val = b[out_idx];
        for (int in_idx = 0; in_idx < in_dim; in_idx++) {
            out_val += in[in_idx] * w[out_idx][in_idx];
        }
        out[out_idx] = out_val;
    }
}


void print_mat(int dim, int chans, float *mat) {
    for(int k = 0; k < chans; k++) {
        printf("Channel %d\n", k);
        for(int i = 0; i < dim; i++) {
            for(int j = 0; j < dim; j++) {
                int idx = i * dim * chans + j * chans + k;
                printf("%f ", mat[idx]);
            }
            printf("\n");
        }
        printf("\n\n");
    }
}

