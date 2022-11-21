import torch
import torch.nn.functional as F
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv_de3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.conv_4 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.conv_de4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.conv_5 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.conv_6 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, padding=1)       
        self.sigmoid = nn.Sigmoid()
        
        self.conv_21 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv_22 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, padding=0)
        self.sigmoid2 = nn.Sigmoid()
        
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)


    def forward(self, x):
        #1
        c_1 = self.act(self.conv_1(x))
        c_p_1 = self.pool_1(c_1)
        c_2 = self.act(self.conv_2(c_p_1))
        c_p_2 = self.pool_2(c_2)
        c_3 = self.act(self.conv_3(c_p_2))
        up_3 = self.conv_de3(c_3)
        cat_23 = torch.cat((up_3, c_2), 1)
        c_4 = self.act(self.conv_4(cat_23))
        up_4 = self.conv_de4(c_4)
        cat_14 = torch.cat((up_4, c_1), 1)
        c_5 = self.act(self.conv_5(cat_14))
        c_6 = self.conv_6(c_5)
        R_out = self.sigmoid(c_6)
        
        c_21 = self.act(self.conv_21(c_1))
        cat_215 = torch.cat((c_21, c_5), 1)
        c_22 = self.conv_22(cat_215)
        L_out = self.sigmoid2(c_22)
        L_out_3 = torch.cat((L_out, L_out, L_out), 1)

        return R_out, L_out_3