# -*- coding: utf-8 -*-
from torch import nn


class NetG(nn.Module):
    """
    生成器定义
    """

    def __init__(self, opt):
        super(NetG, self).__init__()
        ngf = opt.ngf  # 生成器feature map数

        self.main = nn.Sequential(
            # 输入是一个nz维度的噪声，我们可以认为它是一个1*1*nz的feature map
            nn.ConvTranspose2d(opt.nz, ngf * 32, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 32),
            nn.ReLU(True),
            # 上一步的输出形状：(ngf*32) x 4 x 4

            nn.ConvTranspose2d(ngf * 32, ngf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # 上一步的输出形状： (ngf*32) x 8 x 8

            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # 上一步的输出形状： (ngf*8) x 16 x 16

            nn.ConvTranspose2d(ngf * 8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            # 上一步的输出形状：(ngf*4) x 64 x 64

            nn.ConvTranspose2d(ngf * 4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 上一步的输出形状：(ngf*8) x 128 x 128

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 上一步的输出形状：(ngf*8) x 256 x 1256

            nn.ConvTranspose2d(ngf, 3,4, 2, 1, bias=False),
            nn.Tanh()  # 输出范围 -1~1 故而采用Tanh
            # 输出形状：3 x 96 x 96
        )

    def forward(self, input):
        return self.main(input)

class NetD(nn.Module):
    """
    判别器定义
    """

    def __init__(self, opt):
        super(NetD, self).__init__()
        ndf = opt.ndf #ndf=256
        self.main = nn.Sequential(
            # # 输入 3 x 96 x 96
            # nn.Conv2d(3, ndf, 5, 3, 1, bias=False),
            # nn.LeakyReLU(0.2, inplace=True),
            # # 输出 (ndf) x 32 x 32
            # 输入3*256*256
            nn.Conv2d(3,ndf,4,2,1,bias=False),
            nn.LeakyReLU(0.2,inplace=True),
            # ---输出3*128*128

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # ---输出 (ndf*2) x 64 x 64
            # nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 4),
            # nn.LeakyReLU(0.2, inplace=True),
            # 输出 (ndf*4) x 8 x 8

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # ---输出(ndf*4)x32x32
            # nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 8),
            # nn.LeakyReLU(0.2, inplace=True),
            # 输出 (ndf*8) x 4 x 4

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # ---输出(ndf*8)x16x16

            nn.Conv2d(ndf *8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # ---输出(ndf*16)x8x8

            nn.Conv2d(ndf * 16, ndf * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 32),
            nn.LeakyReLU(0.2, inplace=True),
            # ---输出(ndf*32)x4x4

            nn.Conv2d(ndf * 32, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()  # 输出一个数(概率)
        )

    def forward(self, input):
        return self.main(input).view(-1)
