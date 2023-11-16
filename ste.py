import torch
import torch.nn as nn

class SqueezedTokenEnhancer(nn.Module):
    def __init__(self, input_channels, reduction_ratio=4):
        super(SqueezedTokenEnhancer, self).__init__()

        # Depthwise convolution (DWConv3x3)
        self.depthwise_conv = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, groups=input_channels)

        # Squeeze-and-Expansion 1x1 convolutions
        self.squeeze_conv = nn.Conv2d(input_channels, input_channels // reduction_ratio, kernel_size=1)
        self.expand_conv = nn.Conv2d(input_channels // reduction_ratio, input_channels, kernel_size=1)

    def forward(self, x):
        # Depthwise convolution
        dw_conv_output = self.depthwise_conv(x)
        # Squeeze-and-Expansion
        #dw_conv_output = torch.chunk(dw_conv_output, 2, dim=1)
        #print(dw_conv_output[0].shape)
        squeezed = self.squeeze_conv(dw_conv_output)
        expanded = self.expand_conv(squeezed)

        # Residual connection
        output = expanded + x

        return output

# Example usage
batch_size, num_channels, height, width = 1, 64, 32, 32

# Create an instance of SqueezedTokenEnhancer
ste_module = SqueezedTokenEnhancer(input_channels=num_channels)

# Example input tensor
input_tensor = torch.rand((batch_size, num_channels, height, width))

# Forward pass through STE
ste_output = ste_module(input_tensor)

# Print the shape of the output tensor
print("STE Output Tensor Shape:", ste_output.shape)
