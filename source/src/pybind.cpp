#include <torch/extension.h>

#include "../common/pytorch_cpp_helper.hpp"

void linear_forward(    
   Tensor &out,
    const Tensor &in,
    const Tensor &weight,
    const Tensor &bias);



TORCH_LIBRARY(linear, m) {
  m.def("linear_forward", linear_forward);
}