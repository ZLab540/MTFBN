import torch
import torch.nn.functional as F
from torch import nn
from resnext import ResNeXt101
from net_fb import net_fb

class better_upsampling(nn.Module):
      def __init__(self, in_ch, out_ch):
          super(better_upsampling, self).__init__()
          self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=0)

      def forward(self, x,y):
          x = nn.functional.interpolate(x,size= y.size()[2:], mode='nearest', align_corners=None)
          x = F.pad(x, (3 // 2, int(3 / 2), 3 // 2, int(3 / 2)))
          x = self.conv(x)
          return x

class DFEM(nn.Module):
    def __init__(self):
        super(DFEM, self).__init__()
        self.down0 = nn.Sequential(
            nn.Conv2d(256, 64,kernel_size=1), nn.SELU()
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(512, 64,kernel_size=1), nn.SELU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(1024, 64,kernel_size=1), nn.SELU()
        )
        
        self.attention = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=1), nn.SELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(64, 192, kernel_size=1)
        )

        self.conv_fusion = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1), nn.SELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(64, 128, kernel_size=1)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
        )

    def forward(self,x0,x1,x2):
        # Preprocessing
        x0 = self.down0(x0)
        x1 = self.down1(x1)
        x2 = self.down2(x2)

        x1 = F.upsample(x1, size=x0.size()[2:], mode='bilinear')
        x2 = F.upsample(x2, size=x0.size()[2:], mode='bilinear')

        #Strengthening
        n, c, h, w = x0.size()
        concat = torch.cat((x0,x1,x2), 1)
        attention = self.attention(concat)
        attention = attention.view(n, 3, c, h, w)
        attention_out = x0* attention[:, 0, :, :, :] + x1* attention[:, 1, :, :, :]+x2*attention[:, 2, :, :, :]
        attention_out = self.conv1(attention_out)+attention_out

        concat = torch.cat((attention_out,x2), 1)
        gate = self.conv_fusion(concat)
        gate = gate.view(n, 2, c, h, w)
        out = attention_out* gate[:, 0, :, :, :] + x2* gate[:, 1, :, :, :]    
        out = out + self.conv2(out)
        return out


class ours(nn.Module):
    def __init__(self, num_features=128):
        super(ours, self).__init__()
        self.num_features = num_features

        resnext = ResNeXt101()
        self.layer0 = resnext.layer0
        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3

        for m in self.modules():
            if isinstance(m, nn.SELU) or isinstance(m, nn.ReLU):
                m.inplace = True
        self.DFEM = DFEM()
        self.up = better_upsampling(64,32)

        self.p10 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
        )
        
        self.p11 = nn.Sequential(
            nn.Conv2d(num_features//4, num_features//4, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features//4, 3, kernel_size=1)
        )
        
        self.p20 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
        )
        
        self.p21 = nn.Sequential(
            nn.Conv2d(num_features//4, num_features//4, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features//4, 3, kernel_size=1)
        )
        self.attentional_fusion = nn.Sequential(
            nn.Conv2d(num_features // 2, num_features//4, kernel_size=1), nn.SELU(),
            nn.Conv2d(num_features//4,num_features//4, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features//4, num_features//4, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features//4, 6, kernel_size=1)
        )
        self.feedback = net_fb()
    def forward(self, x):
        layer0 = self.layer0(x)  # 1/2
        layer1 = self.layer1(layer0) #1/4
        layer2 = self.layer2(layer1) #1/8
        layer3 = self.layer3(layer2) #1/16

        feature0 = self.DFEM(layer1,layer2,layer3)
        feature0 = self.up(feature0,x)
        f0 = feature0 + self.p10(feature0)

        out1 = self.p11(f0)
        
        f1 = feature0 + self.p20(feature0)
        out2 = x + self.p21(f1)

        n, c, h, w = out2.size()
        attention_phy = torch.cat((f0, f1), 1)
        attention_phy = self.attentional_fusion(attention_phy)
        attention_phy = attention_phy.view(n, 2, c, h, w)
        out3 =  out1* attention_phy[:, 0, :, :, :] + out2 * attention_phy[:, 1, :, :, :]
        feedback = self.feedback(out3,x)
        f0 = f0 + feedback[:,0:32,:,:]
        f1 = f1 + feedback[:,32:,:,:]

        out11 = self.p11(f0)
        out21 = x + self.p21(f1)
        n, c, h, w = out2.size()
        attention_phy = torch.cat((f0, f1), 1)
        attention_phy = self.attentional_fusion(attention_phy)
        attention_phy = attention_phy.view(n, 2, c, h, w)
        out31 = out11*attention_phy[:, 0, :, :, :] + out21 * attention_phy[:, 1, :, :, :]
        return out1,out2,out3,out11,out21,out31