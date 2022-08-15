#include "../common/pytorch_cpp_helper.hpp"
#include "../common/pytorch_device_registry.hpp"
#include <torch/all.h>
#include <torch/python.h>
#include <torch/torch.h>

void linear_forward_impl(
    Tensor &out,
    const Tensor &in,
    const Tensor &weight,
    const Tensor &bias)
{
    DISPATCH_DEVICE_IMPL(linear_forward_impl, out, in, weight, bias);
}

void linear_forward(
     Tensor &out,
    const Tensor &in,
    const Tensor &weight,
    const Tensor &bias)
{
    linear_forward_impl(out, in, weight, bias);
}
