import torch
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot
from torch.autograd import Variable
import math
#import pytorch_colors as colors
import numpy as np

class CSDN_Tem(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CSDN_Tem, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

class FEM(nn.Module):
    def __init__(self, channel):
        super(FEM, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.s_conv1 = nn.Conv2d(channel,channel//4,1)
        self.s_conv2 = nn.Conv2d(channel//4,channel//4,3,1,1)
        self.bn = nn.BatchNorm2d(channel//4)
        self.s_conv3 = nn.Conv2d(channel//4,channel,1)
    def forward(self, x):
        x1 = self.relu(self.s_conv1(x))
        x2 = self.bn(self.s_conv2(x1))
        x3 = self.relu(x2)
        x4 = self.s_conv3(x3)
        x_c = torch.add(x,x4)
        y = self.relu(x_c)
        return y



class enhance_net_nopool(nn.Module):

	def __init__(self):
		super(enhance_net_nopool, self).__init__()

		self.relu = nn.ReLU(inplace=False)

		number_f = 32
		self.e_conv1 = CSDN_Tem(3,number_f)
		self.e_conv2 = FEM(number_f)
		self.e_conv3 = FEM(number_f)
		self.e_conv4 = FEM(number_f)
		self.e_conv5 = CSDN_Tem(number_f*2,number_f) 
		self.e_conv6 = CSDN_Tem(number_f*2,number_f)
		self.e_conv7 = CSDN_Tem(number_f*2,3) 

		self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
		self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)


	def forward(self, x):

		x1 = self.relu(self.e_conv1(x))
		# p1 = self.maxpool(x1)
		x2 = self.relu(self.e_conv2(x1))
		# p2 = self.maxpool(x2)
		x3 = self.relu(self.e_conv3(x2))
		# p3 = self.maxpool(x3)
		x4 = self.relu(self.e_conv4(x3))

		x5 = self.relu(self.e_conv5(torch.cat([x3,x4],1)))
		# x5 = self.upsample(x5)
		x6 = self.relu(self.e_conv6(torch.cat([x2,x5],1)))

		x_r = F.tanh(self.e_conv7(torch.cat([x1,x6],1)))


		x = x + x_r*(torch.pow(x,2)-x)
		x = x + x_r*(torch.pow(x,2)-x)
		x = x + x_r*(torch.pow(x,2)-x)
		x = x + x_r*(torch.pow(x,2)-x)
		x = x + x_r*(torch.pow(x,2)-x)
		x = x + x_r*(torch.pow(x,2)-x)
		x = x + x_r*(torch.pow(x,2)-x)
		enhance_image = x + x_r*(torch.pow(x,2)-x)
		return enhance_image,x_r



if __name__ == "__main__":
	net = enhance_net_nopool()
	x = torch.zeros(1, 3, 500, 500, dtype=torch.float, requires_grad=False)
	out = net(x)
	g = make_dot(out)
	g.render('3_model', view=True)




