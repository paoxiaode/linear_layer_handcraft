#include "../../common/pytorch_cpp_helper.hpp"
#include "../../common/pytorch_device_registry.hpp"

void LinearCUDAKernelLauncher(
    Tensor &out,
    const Tensor &in,
    const Tensor &weight,
    const Tensor &bias);


void linear_forward_cuda(
    Tensor &out,
    const Tensor &in,
    const Tensor &weight,
    const Tensor &bias){
        LinearCUDAKernelLauncher(out, in, weight, bias);
    }

void linear_forward_impl(    
    Tensor &out,
    const Tensor &in,
    const Tensor &weight,
    const Tensor &bias);

REGISTER_DEVICE_IMPL(linear_forward_impl, CUDA,
                     linear_forward_cuda);