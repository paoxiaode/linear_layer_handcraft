import torch
import torch.nn as nn
import os
ops_library_abs_path = os.path.abspath("./source/build/liblinear.so")
torch.ops.load_library(ops_library_abs_path)
class linear_relu(nn.Module):
    def __init__(self, in_dim:int, out_dim:int) -> None:
        super(linear_relu, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.act = nn.ReLU()

    def forward(self, h):
        h = self.linear(h)
        return self.act(h)

class linear_relu_naive():
    def __init__(self, M, K, N, weight, bias) -> None:
        self.M = M
        self.K = K
        self.N = N
        self.weight = weight
        self.bias = bias

    def linear(self, input):
        output = torch.zeros([self.M,self.N])
        for i in range(self.M):
            for k in range(self.K):
                for j in range(self.N):
                    if k == 0:
                        output[i,j] = self.bias[j]
                    output[i,j] += input[i,k] * self.weight[k,j]
        return output
    
    def relu(self, input):
        for i in range(self.M):
            for j in range(self.N):
                input[i,j] = max(input[i,j], 0)
        return input

    def __call__(self, input):
        input = self.linear(input)
        return input

class linear_relu_op(torch.autograd.Function):
    def forward(self, input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor ):
        output = torch.zeros([input.shape[0],weight.shape[1]]).to("cuda:0")
        weight_cu = weight.to("cuda:0")
        input_cu = input.to("cuda:0")
        bias_cu = bias.to("cuda:0")
        torch.ops.linear.linear_forward(output, input_cu, weight_cu, bias_cu)
        return output

def main():
    in_dim = 200
    out_dim = 100
    M = 10
    ###############################################
    module_torch = linear_relu(in_dim, out_dim)
    input = torch.rand([M, in_dim])
    output_torch = module_torch(input)
    print(output_torch.shape)
    print(output_torch.dtype)

    ##############################################
    weight = torch.rand([in_dim, out_dim])
    bias = torch.rand([out_dim])
    module_naive = linear_relu_naive(M, in_dim, out_dim, weight, bias)
    output_naive = module_naive(input)
    print(output_naive.shape)
    #############################################
    output_op = linear_relu_op.apply(input, weight, bias)
    print(output_op.shape)
    print(torch.isclose(output_naive.to("cuda:0"), output_op))

if __name__ == "__main__":
    main()