from statistics import mode
import torch
import torch.nn as nn
import torchvision.models as models
from ResNet import ResNet50
import torch.nn.functional as F
from einops import rearrange, repeat

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interpolate = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
    def forward(self, x):
        x = self.interpolate(x, scale_factor=self.scale_factor, mode=self.mode,align_corners=True)
        return x


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=1, bias=False)
    def forward(self, ftr):
        ftr_avg = torch.mean(ftr, dim=1, keepdim=True)
        ftr_max, _ = torch.max(ftr, dim=1, keepdim=True)
        ftr_cat = torch.cat([ftr_avg, ftr_max], dim=1)
        att_map = F.sigmoid(self.conv(ftr_cat))
        return att_map

def convblock(in_, out_, ks, st, pad):
    return nn.Sequential(
        nn.Conv2d(in_, out_, ks, st, pad),
        nn.BatchNorm2d(out_),
        nn.ReLU(inplace=True)
    )


class Decoder(nn.Module):
    def __init__(self, in_1, in_2):
        super(Decoder, self).__init__()
        self.conv1 = convblock(in_1, 128, 3, 1, 1)
        self.conv_out = convblock(128, in_2, 3, 1, 1)

    def forward(self, pre,cur):
        cur_size = cur.size()[2:]
        pre = self.conv1(F.interpolate(pre, cur_size, mode='bilinear', align_corners=True))
        fus = pre
        return self.conv_out(fus)

class CA(nn.Module):
    def __init__(self,in_ch):
        super(CA, self).__init__()
        self.avg_weight = nn.AdaptiveAvgPool2d(1)
        self.max_weight = nn.AdaptiveMaxPool2d(1)
        self.fus = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(in_ch // 2, in_ch, 1, 1, 0),
        )
        self.c_mask = nn.Sigmoid()
    def forward(self, x):
        avg_map_c = self.avg_weight(x)
        max_map_c = self.max_weight(x)
        c_mask = self.c_mask(torch.add(self.fus(avg_map_c), self.fus(max_map_c)))
        return torch.mul(x, c_mask)

class FinalOut(nn.Module):
    def __init__(self):
        super(FinalOut, self).__init__()
        self.ca =CA(128)
        self.score = nn.Conv2d(128, 1, 1, 1, 0)
    def forward(self,f1,f2,xsize):
        f1 = torch.cat((f1,f2),1)
        f1 = self.ca(f1)
        score = F.interpolate(self.score(f1), xsize, mode='bilinear', align_corners=True)
        return score

class SaliencyNet(nn.Module):
    def __init__(self):
        super(SaliencyNet, self).__init__()
        self.c4=nn.Conv2d(2048*2,2048,kernel_size=1)
        self.c3=nn.Conv2d(1024*2, 1024, kernel_size=1)
        self.c2=nn.Conv2d(512*2,512,kernel_size=1)
        self.c1 = nn.Conv2d(256*2, 256, kernel_size=1)
        self.c = nn.Conv2d(64*2, 64, kernel_size=1)
        self.spa = SpatialAttention()
        self.ca4 = CA(2048*2)
        self.ca3 = CA(2048)
        self.ca2 = CA(1024)
        self.ca1 = CA(512)
        self.ca = CA(128)

        self.d4_r = Decoder(2048,1024)
        self.d3_r= Decoder(1024,512)
        self.d2_r= Decoder(512,256)
        self.d1_r = Decoder(256, 64)

        self.score = nn.Conv2d(128, 1, 1, 1, 0)
        self.score4 = nn.Conv2d(1024, 1, 1, 1, 0)
        self.score3 = nn.Conv2d(512, 1, 1, 1, 0)
        self.score2 = nn.Conv2d(256, 1, 1, 1, 0)
        self.score1 = nn.Conv2d(64, 1, 1, 1, 0)

    def forward(self,tt,r,r1,r2,r3,r4,t4,alpha,t,t1,t2,t3,aaa):
        xsize=tt.size()[2:]
        alpha5 = repeat(alpha, 'b n  -> b n h w', h=int(aaa/32), w=int(aaa/32))
        tt4 = torch.mul(t4, 1 - alpha5)
        sp5 = self.spa(tt4)
        temp = r4.mul(sp5)
        r4 = r4 + temp
        d4=r4.mul(alpha5)+t4.mul(1-alpha5)
        d4 = torch.cat((d4, r4), 1)
        d4=self.ca4(d4)
        d4 = self.c4(d4)
        d4=self.d4_r(d4,r3)
        u4=d4

        alpha4 = repeat(alpha, 'b n  -> b n h w', h=int(aaa/16), w=int(aaa/16))
        tt3 = torch.mul(t3, 1 - alpha4)
        sp4 = self.spa(tt3)
        temp = d4.mul(sp4)
        d4=d4+temp

        d3 = r3.mul(alpha4) + t3.mul(1 - alpha4)
        d3 = torch.cat((d4, d3), 1)
        d3=self.ca3(d3)
        d3=self.c3(d3)
        d3 = self.d3_r(d3, r2)
        u3=d3

        alpha3 = repeat(alpha, 'b n  -> b n h w', h=int(aaa/8), w=int(aaa/8))
        tt2 = torch.mul(t2, 1 - alpha3)
        sp3 = self.spa(tt2)
        temp = d3.mul(sp3)
        d3 = d3 + temp

        d2 = r2.mul(alpha3) + t2.mul(1 - alpha3)
        d2 = torch.cat((d3, d2), 1)
        d2=self.ca2(d2)
        d2 = self.c2(d2)
        d2 = self.d2_r(d2, r1)
        u2=d2

        alpha2 = repeat(alpha, 'b n  -> b n h w', h=int(aaa/4), w=int(aaa/4))
        tt1 = torch.mul(t1, 1 - alpha2)
        sp2 = self.spa(tt1)
        temp = d2.mul(sp2)
        d2 = d2 + temp

        d1 = r1.mul(alpha2) + t1.mul(1 - alpha2)
        d1 = torch.cat((d2, d1), 1)
        d1=self.ca1(d1)
        d1 = self.c1(d1)
        d1 = self.d1_r(d1, r)
        u1=d1


        alpha1 = repeat(alpha, 'b n  -> b n h w', h=int(aaa/4), w=int(aaa/4))
        tt = torch.mul(t, 1 - alpha1)
        sp1 = self.spa(tt)
        temp = d1.mul(sp1)
        d1 = d1 + temp

        d = r.mul(alpha1) + t.mul(1 - alpha1)
        d = torch.cat((d, d1), 1)
        d=self.ca(d)
        result = F.interpolate(self.score(d), xsize, mode='bilinear', align_corners=True)
        u4=F.interpolate(self.score4(u4), xsize, mode='bilinear', align_corners=True)
        u3 = F.interpolate(self.score3(u3), xsize, mode='bilinear', align_corners=True)
        u2 = F.interpolate(self.score2(u2), xsize, mode='bilinear', align_corners=True)
        u1 = F.interpolate(self.score1(u1), xsize, mode='bilinear', align_corners=True)

        return result,u4,u3,u2,u1

#baseline
class Baseline(nn.Module):
    def __init__(self,channel=32):
        super(Baseline, self).__init__()

        #Backbone model
        self.resnet = ResNet50('rgb')
        self.resnet_t = ResNet50('rgbt')
        self.s_net = SaliencyNet()

        self.cc = nn.Conv2d(2048, 1, kernel_size=1)
        self.cc1 = nn.ConvTranspose2d(1, 1, kernel_size=16, padding=4, stride=8)
        self.cc2 = nn.ConvTranspose2d(1, 1, kernel_size=16, padding=4, stride=8)
        self.cc3 = nn.ConvTranspose2d(1, 1, kernel_size=8, padding=2, stride=4)
        self.cc4 = nn.ConvTranspose2d(1, 1, kernel_size=4, padding=1, stride=2)
        self.chutu = nn.ConvTranspose2d(1, 1, kernel_size=64, padding=16, stride=32)
        self.sigmoid = nn.Sigmoid()

        self.gap=nn.AdaptiveAvgPool2d((1, 1))
        self.flatten=nn.Flatten()
        self.fc=nn.Linear(3*1*1,1)

        self.cl=nn.Conv2d(3,1,kernel_size=1)
        self.cl1=nn.Conv2d(1,1,kernel_size=7,stride=2,padding=3)

        if self.training:
            self.initialize_weights()

    def forward(self,x,x_t, L):
        aaa=x.size(-1)
        aaa=int(aaa)
        alpha=self.gap(L)
        alpha=self.flatten(alpha)
        alpha=self.fc(alpha)
        alpha=self.sigmoid(alpha)
        #RGB
        #conv1
        tt=x
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        #conv2
        x1 = self.resnet.layer1(x)

        #conv3
        x2 = self.resnet.layer2(x1)

        #conv4
        x3 = self.resnet.layer3(x2)

        #conv5
        x4 = self.resnet.layer4(x3)

        #T
        x_t = self.resnet_t.conv1(x_t)
        x_t = self.resnet_t.bn1(x_t)
        x_t = self.resnet_t.relu(x_t)
        x_t = self.resnet_t.maxpool(x_t)


        ttt = self.cc(x4)
        tt1 = self.cc1(ttt)
        tt1 = self.sigmoid(tt1)
        alpha1=repeat(alpha, 'b n  -> b n h w', h=int(aaa/4),w=int(aaa/4))
        tt1=torch.mul(tt1,alpha1)
        temp = x_t.mul(tt1)
        x_t = x_t+temp

        x_t1 = self.resnet_t.layer1(x_t)
        tt2 = self.cc2(ttt)
        tt2 = self.sigmoid(tt2)
        alpha2 = repeat(alpha, 'b n  -> b n h w', h=int(aaa/4), w=int(aaa/4))
        tt2 =tt2.mul(alpha2)
        temp = x_t1.mul(tt2)
        x_t1 = x_t1 + temp

        x_t2 = self.resnet_t.layer2(x_t1)
        tt3 = self.cc3(ttt)
        tt3 = self.sigmoid(tt3)
        alpha3 = repeat(alpha, 'b n  -> b n h w', h=int(aaa/8), w=int(aaa/8))
        tt3 =tt3.mul(alpha3)
        temp = x_t2.mul(tt3)
        x_t2 = x_t2 + temp

        x_t3 = self.resnet_t.layer3(x_t2)
        tt4 = self.cc4(ttt)
        tt4 = self.sigmoid(tt4)
        alpha4 = repeat(alpha, 'b n  -> b n h w', h=int(aaa/16), w=int(aaa/16))
        tt4 = tt4.mul(alpha4)
        temp = x_t3.mul(tt4)
        x_t3 = x_t3 + temp

        x_t4 = self.resnet_t.layer4(x_t3)
        tt5 = self.sigmoid(ttt)
        alpha5 = repeat(alpha, 'b n  -> b n h w', h=int(aaa/32), w=int(aaa/32))
        tt5 = tt5.mul(alpha5)
        temp = x_t4.mul(tt5)
        x_t4 = x_t4 + temp

        #Decoder
        result_r,u4,u3,u2,u1 = self.s_net(tt,x,x1,x2,x3,x4, x_t4,alpha,x_t,x_t1,x_t2,x_t3,aaa)
        result_r=self.sigmoid(result_r)
        u4=self.sigmoid(u4)
        u3 = self.sigmoid(u3)
        u2 = self.sigmoid(u2)
        u1 = self.sigmoid(u1)
        ttt = self.chutu(ttt)
        ttt = self.sigmoid(ttt)
        return result_r,ttt,u4,u3,u2,u1

        #initialize the weights
    def initialize_weights(self):
        res50 = models.resnet50(pretrained=True)
        pretrained_dict = res50.state_dict()
        all_params = {}
        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())
        self.resnet.load_state_dict(all_params)

        all_params = {}
        for k, v in self.resnet_t.state_dict().items():
            if k=='conv1.weight':
                all_params[k]=torch.nn.init.normal_(v, mean=0, std=1)
            elif k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet_t.state_dict().keys())
        self.resnet_t.load_state_dict(all_params)







