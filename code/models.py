import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, padding_mode='zeros', bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   groups=in_channels, stride=stride, bias=bias,
                                   padding=kernel_size//2, padding_mode=padding_mode)

        self.pointwise = nn.Conv2d(in_channels, out_channels, 
                                   kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

#################### ConvNet ####################

class ConvNet(nn.Module):
  """
  Simple convolutional model.
  """
  def __init__(self,
               filter_name,
               in_channels: int,
               out_channels: list,
               kernel_size: list,
               stride: list,
               padding_mode: str='zeros',
               separable_conv=False,
               threshold: float=0.5):
        
      super().__init__()
      self.filter_name = filter_name
      self.conv = SeparableConv2d if separable_conv else nn.Conv2d
      self.threshold = threshold

      self.layers = []
      self.layers.extend([self.conv(in_channels=in_channels,
                               out_channels=out_channels[0],
                               kernel_size=kernel_size[0],
                               stride=stride[0],
                               padding=kernel_size[0]//2, 
                               padding_mode=padding_mode),
                    nn.ReLU()])

      for i in range(1, len(out_channels)-1):
        self.layers.extend([self.conv(in_channels=out_channels[i-1],
                               out_channels=out_channels[i],
                               kernel_size=kernel_size[0],
                               stride=stride[0],
                               padding=kernel_size[0]//2, 
                               padding_mode=padding_mode),
                            nn.ReLU()])

      self.layers.extend([self.conv(in_channels=out_channels[-2],
                               out_channels=out_channels[-1],
                               kernel_size=kernel_size[0],
                               stride=stride[0],
                               padding=kernel_size[0]//2, 
                               padding_mode=padding_mode)])
                          
      self.model = nn.Sequential(*self.layers)
    

  def forward(self, x):
      logits = self.model(x.float()) 
      probas = torch.sigmoid(logits)
      if self.filter_name in ['canny', 'niblack']:
        y_hat = (probas>=self.threshold).float()
      else:
        y_hat = probas
      return probas, y_hat


#################### ResNet ####################

class StartBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
            
        self.start_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3, stride=1, bias=False),
            nn.BatchNorm2d(out_channels))

    def forward(self, x):
        out = F.leaky_relu_(self.start_block(x))
        return out
    
    
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel1, kernel2, is_last):
        super().__init__()
        self.kernel1 = kernel1
        self.kernel2 = kernel2
        self.is_last = is_last
        self.out_channels = 1 if self.is_last else in_channels
        
        self.basic_block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel1, padding=self.kernel1//2, stride=1, bias=False),
            nn.BatchNorm2d(out_channels))
        
        self.basic_block2 = nn.Sequential(
            nn.Conv2d(out_channels, self.out_channels, kernel_size=self.kernel2, padding=self.kernel2//2, stride=1, bias=False),
            nn.BatchNorm2d(self.out_channels))
        

    def forward(self, x):
        identity = x.clone()
        out = F.leaky_relu_(self.basic_block1(x))
        out = self.basic_block2(out)
        
        if not self.is_last:
            out += identity
            out = F.leaky_relu_(out)
        else:
            out = torch.sigmoid(out)
        return out
    
    
class ResNet(nn.Module):
    def __init__(self, filter_name, in_channels, mid_channels, threshold=0.5):
        super().__init__()
        self.filter_name = filter_name
        self.threshold = threshold

        self.model = nn.Sequential(
            StartBlock(in_channels, mid_channels), #1 
            BasicBlock(mid_channels, mid_channels, 5, 3, False),   #3
            BasicBlock(mid_channels, mid_channels, 5, 3, False),   #5
            BasicBlock(mid_channels, mid_channels, 5, 3, False),   #7
            BasicBlock(mid_channels, mid_channels, 5, 3, False),   #9
            BasicBlock(mid_channels, mid_channels, 5, 3, False),   #11
            BasicBlock(mid_channels, mid_channels, 5, 3, True)     #13
        )

    def forward(self, x):
        probas = self.model(x)
        if self.filter_name in ['canny', 'niblack']:
            y_hat = (probas>=self.threshold).float()
        else:
            y_hat = probas
        return probas, y_hat 
    
    
#################### UNet ####################


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

 
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

    
class UNet(nn.Module):
    def __init__(self, filter_name, in_channels, mid_channels, bilinear=True, threshold=0.5):
        super(UNet, self).__init__()
        self.filter_name = filter_name
        self.threshold = threshold

        self.inc = DoubleConv(in_channels, mid_channels)
        self.down1 = Down(mid_channels, 2*mid_channels)
        self.down2 = Down(2*mid_channels, 4*mid_channels)
        self.down3 = Down(4*mid_channels, 4*mid_channels)
        factor = 2 if bilinear else 1
        self.down4 = Down(4*mid_channels, 4*mid_channels)
        self.up1 = Up(8*mid_channels, 8*mid_channels // factor, bilinear)
        self.up2 = Up(8*mid_channels, 4*mid_channels // factor, bilinear)
        self.up3 = Up(4*mid_channels, 2*mid_channels // factor, bilinear)
        self.up4 = Up(2*mid_channels, mid_channels, bilinear)
        self.outc = OutConv(mid_channels, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        probas = torch.sigmoid(x)
        if self.filter_name in ['canny', 'niblack']:
          y_hat = (probas>=self.threshold).float()
        else:
          y_hat = probas
        return probas, y_hat    
    
    
# class UNet(nn.Module):
#     def __init__(self, filter_name, in_channels, mid_channels, bilinear=True, threshold=0.5):
#         super(UNet, self).__init__()
#         self.filter_name = filter_name
#         self.threshold = threshold

#         self.inc = DoubleConv(in_channels, mid_channels)
#         self.down1 = Down(mid_channels, 2*mid_channels)
#         self.down2 = Down(2*mid_channels, 4*mid_channels)
#         self.down3 = Down(4*mid_channels, 8*mid_channels)
#         factor = 2 if bilinear else 1
#         self.down4 = Down(8*mid_channels, 16*mid_channels // factor)
#         self.up1 = Up(16*mid_channels, 8*mid_channels // factor, bilinear)
#         self.up2 = Up(8*mid_channels, 4*mid_channels // factor, bilinear)
#         self.up3 = Up(4*mid_channels, 2*mid_channels // factor, bilinear)
#         self.up4 = Up(2*mid_channels, mid_channels, bilinear)
#         self.outc = OutConv(mid_channels, 1)

#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         x = self.outc(x)
#         probas = torch.sigmoid(x)
#         if self.filter_name in ['canny', 'niblack']:
#           y_hat = (probas>=self.threshold).float()
#         else:
#           y_hat = probas
#         return probas, y_hat