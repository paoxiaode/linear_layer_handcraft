#include "../include/gemm.cuh"
#include "../../common/pytoch_cuda_hepler.hpp"
#include <iostream>

void LinearCUDAKernelLauncher(
    Tensor &out,
    const Tensor &in,
    const Tensor &weight,
    const Tensor &bias)
{
    const int64_t M = in.size(0);
    const int64_t K = in.size(1);
    const int64_t N = weight.size(1);
    if (M < 0){
        return;
    }
    at::cuda::CUDAGuard device_guard(out.device());
    dim3 threads_per_block(128);
    dim3 blocks_per_grid(M);
    const int TILE_SIZE = (N + 127)/128;
    linear_float<<<blocks_per_grid,threads_per_block>>>(out.data_ptr<float>(), in.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), M, K, N, TILE_SIZE);

    AT_CUDA_CHECK(cudaGetLastError());
}