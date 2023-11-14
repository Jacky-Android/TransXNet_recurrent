import torch
import torch.nn as nn
import torch.nn.functional as F

class IDConv2d(nn.Module):
    def __init__(self, in_channels, kernel_size, groups):
        super(IDConv2d, self).__init__()
        self.groups = groups
        

        # Adaptive average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=kernel_size)

        # Sequential 1x1 convolutions
        self.conv1 = nn.Conv2d(in_channels, in_channels//groups , kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels//groups, groups * in_channels, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels*2, in_channels, kernel_size=1, bias=False)

        # Learnable parameters
        self.weights = nn.Parameter(torch.rand(groups,in_channels, kernel_size,kernel_size))
        self.kernel_size = kernel_size

    def forward(self, x):
        batch_size, num_channels, height, width = x.size()
        shotcut = x
        # Adaptive average pooling
        x_avg = self.avg_pool(x)
        
        a_prime = self.conv2(self.conv1(x_avg))
        a_prime = a_prime.view(batch_size, self.groups, num_channels,self.kernel_size,self.kernel_size)
        a_prime = F.softmax(a_prime,dim=1)
        #print(self.weights.unsqueeze(0).shape)
        a = a_prime*self.weights.unsqueeze(0)
        w = a.sum(dim=1)
        w = F.interpolate(w, size=(height, width), mode='bilinear', align_corners=True)
        w = torch.cat([shotcut,w],dim=1)
        w = self.conv3(w)
        #x_avg = self.conv1(x_avg)
        # Sequential 1x1 convolutions
        

        return w
