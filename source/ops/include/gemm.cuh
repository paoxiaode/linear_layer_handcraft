#ifndef GEMM_CUH
#define GEMM_CUH

#include <stdio.h>
#include <cuda.h>

__global__ void linear_float(
    float *out, 
    const float *in, 
    const float *weight, 
    const float *bias,
    const int64_t M,
    const int64_t K,
    const int64_t N,
    const int TILE_SIZE) 
{
    int tidx = threadIdx.x;
    int bidx = blockIdx.x;
    int start_idx_y = tidx * TILE_SIZE;
    int out_num = 0;
    float tmp;
    // out[bidx][tidx*TILE_SIZE+out_num] = in[bidx][k] * weight[k][tidx*TILE_SIZE+out_num] + bias[tidx*TILE_SIZE+out_num]
    while (out_num < TILE_SIZE) {
        if (start_idx_y + out_num < N){
            tmp = bias[start_idx_y + out_num];
            for(int k = 0; k < K; k++){
                tmp += in[bidx * K + k] * weight[k * N + start_idx_y +out_num];
            }
            out[bidx * N + start_idx_y + out_num] = tmp;
        }
        else{
            break;
        }
        out_num += 1;
    }
}


#endif