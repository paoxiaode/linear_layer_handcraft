import torch
import torch.nn as nn
import os
ops_library_abs_path = os.path.abspath("./source/build/liboc20_customized_ops.so")
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
        return self.relu(input)

def main():
    in_dim = 64
    out_dim = 128
    M = 100
    ###############################################
    module_torch = linear_relu(in_dim, out_dim)
    input = torch.rand([M, in_dim])
    output_torch = module_torch(input)
    print(output_torch.shape)
    ##############################################
    weight = torch.rand([in_dim, out_dim])
    bias = torch.rand([out_dim])
    module_naive = linear_relu_naive(M, in_dim, out_dim, weight, bias)
    output_naive = module_naive(input)
    print(output_naive.shape)
    #############################################

if __name__ == "__main__":
    main()