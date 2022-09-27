# linear_layer_handcraft

Through this repo, we can define our custom op (eg. linear) in CUDA, then bind it to C++ and use it in pytorch.

## Code process

* First define the kernel

  ```cpp
  __global__ void linear_float(…)
  ```
* Define the launch dimension and
  launch function

  ```cpp
  void LinearCUDAKernelLauncher(
      Tensor &out,
      const Tensor &in,
      const Tensor &weight,
      const Tensor &bias)
  {
      dim3 threads_per_block(128);
      dim3 blocks_per_grid(M);
      const int TILE_SIZE = (N + 127)/128;
      linear_float<<<blocks_per_grid,threads_per_block>>>(…);
  }
  ```
* Bind the launch function to torch
  extension

  ```cpp
  #include <torch/extension.h>TORCH_LIBRARY(linear, m) {
    m.def("linear_forward", linear_forward);
  }
  ```
* Call the custom op by torch.ops

  ```cpp
  torch.ops.linear.linear_forward(output, input_cu, weight_cu, bias_cu)
  ```

## Bind op into pytorch

[Extending TorchScript with Custom C++ Operators](https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html)


[paoxiaode/linear_layer_handcraft:手动实现linear层的前向](https://github.com/paoxiaode/linear_layer_handcraft)
## CUDA learning

[CUDA programming guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)

[CUDA C++ best practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)

[CUDA samples](https://github.com/NVIDIA/cuda-samples)
