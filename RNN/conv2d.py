# 2d conv
import torch
import torch.nn.functional as F
import torch.nn as nn

def conv2d(x,kernel,pad=1,stride=1):
    
    in_h,in_w = x.shape
    k_h,k_w = kernel.shape
    out_h = 1 + (in_h + pad*2 - k_w) // stride
    out_w = 1 + (in_w + pad*2 - k_h) // stride
    
    x = F.pad(x,(pad,pad,pad,pad))
    out = torch.zeros((out_h,out_w))
    for i in range(0,out_h):
        for j in range(0,out_w):
            h_start = i * stride
            w_start = j * stride
            out_block = torch.sum(x[h_start : h_start+k_h,w_start : w_start+k_w] * kernel)
            out[i,j] = out_block
    return out

class Conv2d_scratch:
    def __init__(
            self,
            in_channels:int,
            out_channels:int,
            kernel_size:int|tuple[int,int],
            stride:int,
            padding:int,
            ) -> None:  # python约定:init必须返回None
        
        self.in_channel,self.out_channel,self.kernel_size,self.stride,self.padding = \
        in_channels,out_channels,kernel_size,stride,padding
        pass
    
    def __call__(self,x:torch.Tensor) -> torch.Tensor:
        return self.forward(x)
    
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        "Args:  x.shape=[N,in_channel,H,W]"
        "Returns:   y.shape(N,out_channel,H_out,W_out),out_channel:num of kernels"

        N,C,H,W = x.shape
        assert  C == self.in_channel
        kernel = torch.randn(self.kernel_size)
        k_h,k_w = kernel.shape
        
        weight = torch.randn((self.out_channel,self.in_channel,k_h,k_w))

        out_h,out_w = 1 + (H + self.padding*2 - k_h)//self.stride,1 + (W + self.padding*2 - k_w)//self.stride
        out = torch.zeros((N,self.out_channel,out_h,out_w))

        for n in range(N):
            for oc in range(self.out_channel):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        x_block = x[n,:,h_start:h_start+k_h,w_start:w_start+k_w]
                        out[n,oc,i,j] = torch.sum(x_block * weight[oc])
        return out

if __name__ == "__main__":
    out_1 = conv2d(x = torch.rand((3,4)),kernel = torch.rand((2,2)),pad = 1,stride = 2)

    conv2d_scatch = Conv2d_scratch(in_channels=3,out_channels=16,kernel_size=(3,3),stride=1,padding=0)
    out_2 = conv2d_scatch(x = torch.randn((4,3,4,3)))

    conv2d_torch = nn.Conv2d(in_channels=3,out_channels=16,kernel_size=(3,3),stride=1,padding=0)
    out_3 = conv2d_torch(torch.randn((4,3,4,3)))
    pass


