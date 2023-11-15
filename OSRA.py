import torch
import torch.nn as nn
import torch.nn.functional as F

class SRA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
    
class OSRA(nn.Module):
    def __init__(self, input_channels, dim, num_attention_heads):
        super(OSRA, self).__init__()

        # Depthwise separable convolution for OSR
        self.osr = SRA(input_channels, num_attention_heads)
        self.num_heads = num_attention_heads
        # Linear layer for Q
        self.linear_q = nn.Linear(input_channels, dim)

        # Linear layer for K and V
        self.linear_kv = nn.Linear(input_channels, 2 * dim)

        # Relative position bias matrix
        self.relative_bias = nn.Parameter(torch.rand((1, num_attention_heads, 1, 1)))
        self.depthconv = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, groups=input_channels)

    def forward(self, x):
        # OSR convolution
        b,c,h,w = x.shape
        #x = x.permute(0, 2, 3, 1)
        x = x.flatten(2).transpose(1, 2)
        #where LR(·) denotes a local refinement module that is instantiated by a 3×3 depthwise convolution
        y = self.osr(x,h,w)
        osr_output = y.permute(0, 2, 1).reshape(b,c,h,w)
        osr_output = self.depthconv(osr_output).reshape(b, c, -1).permute(0, 2, 1)

        # Linear layers for Q, K, and V
        q = self.linear_q(x).reshape(b, -1, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        x = self.linear_kv(osr_output+y).reshape(b, -1, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        
        k,v = x[0],x[1]
        
        # Softmax attention
        d = q.size(-1) ** 0.5
        attention_scores = F.softmax(q @ k.transpose(-2, -1) / d + self.relative_bias, dim=-1)
        output = attention_scores @ v
        output = output.transpose(1, 2).reshape(b,h*w,c)
        output = output.reshape(b,h,w,c).permute(0, 3, 1, 2)

        return output

# Example usage
batch_size, num_channels, height, width = 1, 64, 32, 32
num_attention_heads = 4

# Create an instance of OSRA
osra_layer = OSRA(input_channels=num_channels,dim=num_channels, num_attention_heads=num_attention_heads)

# Example input tensor
input_tensor = torch.rand((batch_size, num_channels, height, width))

# Forward pass through OSRA
osra_output = osra_layer(input_tensor)

# Print the shape of the output tensor
print("OSRA Output Tensor Shape:", osra_output.shape)
