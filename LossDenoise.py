import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torchvision
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import imageio
import matplotlib.image as pm
import cv2
from scipy.signal import convolve2d
from torchvision import transforms
from math import exp
from torchvision.transforms import InterpolationMode
from torch.autograd import Variable


class perception_loss(nn.Module):
    def __init__(self):
        super(perception_loss, self).__init__()
        features = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES).features
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        # out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return h_relu_2_2


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    transform = torchvision.transforms.RandomResizedCrop(256, antialias=True, interpolation=InterpolationMode.BILINEAR)
    transform1 = torchvision.transforms.CenterCrop(224)
    transform2 = torchvision.transforms.Normalize(torch.mean(img1), torch.std(img1, unbiased=False))
    transform3 = torchvision.transforms.Normalize(torch.mean(img2), torch.std(img2, unbiased=False))
    img1 = transform2(transform1(transform(img1)))
    img2 = transform3(transform1(transform(img2)))
    A = perception_loss().cuda()
    img1 = A(img1)
    img2 = A(img2)
    img1 = img1.mean(0).mean(0)
    img2 = img2.mean(0).mean(0)
    img1 = torch.unsqueeze(img1, 0)
    img2 = torch.unsqueeze(img2, 0)
    img1 = torch.cat([img1,img1,img1],dim=0)
    img2 = torch.cat([img2,img2,img2],dim=0)


    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    ( channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

class noisy_loss():

    def convolution(self,conv_input):
        conv_input = torch.unsqueeze(conv_input,0)
        #print("conv",conv_input.shape)
        c = torch.nn.Conv2d(1, 1, (3, 3), stride=2, padding=1, bias=False)
        c.weight.data = torch.Tensor([[[[0, -1, 0],
                                [-1, 4, -1],
                                [0, -1, 0]]]]).cuda()
        convolved_image = c(conv_input)
        return convolved_image

    def otsu(self,image):
        mean = torch.mean(image)
        mask_less = (image<mean)
        mask_lager = (image>mean)
        image_less = torch.masked_select(image,mask_less)
        image_lager = torch.masked_select(image,mask_lager)
        omega0 = image_less.numel() / image.numel()
        omega1 = image_lager.numel() / image.numel()
        mu0 = torch.mean(image_less)
        mu1 = torch.mean(image_lager)
        g = omega0 * omega1 * (mu0 - mu1) ** 2
        return g
    def binary(self,binary_input,w):
        binary_output = torch.where(binary_input >= w, 1, 0)
        #print(binary_output)
        return binary_output

    def loss_noisy(self):
        mean_original = torch.mean(self.binary_original)
        mean_final = torch.mean(self.binary_final)
        loss_noise = 1 - (torch.minimum(mean_final,mean_original)/torch.maximum(mean_final,mean_original))
        #print("noisy",loss_noise)
        return loss_noise

    def __init__(self,original_image,final_image):
        self.original_image = original_image
        self.final_image = final_image

        self.original_image = torch.mean(self.original_image, 0)
        self.final_image = torch.mean(self.final_image, 0)

        w_original = self.otsu(self.original_image)
        w_final = self.otsu(self.final_image)
        
        self.original_image = self.convolution(self.original_image)
        self.final_image = self.convolution(self.final_image)
        #print("shape",self.original_image.shape,self.final_image.shape)

        self.binary_original = self.binary(self.original_image,w_original).float()
        self.binary_final = self.binary(self.final_image,w_final).float()



class Ssim_Loss(nn.Module):
    def __init__(self):
        super(Ssim_Loss, self).__init__()

    def forward(self,x,y):
        x = torch.mean(x,dim=0)
        y = torch.mean(y,dim=0)
        #print(x.shape,y.shape)
        #Ssim = ssim_loss(x,y).loss_ssim()
        #return Ssim
class La_Loss(nn.Module):
    def __init__(self,):
        super(La_Loss, self).__init__()

    def forward(self,x,y):
        x = torch.mean(x, dim=0)
        y = torch.mean(y, dim=0)
        return noisy_loss(x,y).loss_noisy()










if __name__ == '__main__':

    #test part(start)
    img1 = Image.open('./data/result/DICM/01.jpg')
    img1 = np.array(img1)
    img1 = torch.FloatTensor(img1).cuda()
    img1 = img1.permute(2 ,0 ,1)



    img2 = Image.open('./data/test_data/DICM/01.jpg')
    img2 = np.array(img2)
    img2 = torch.FloatTensor(img2).cuda()
    img2 = img2.permute(2, 0, 1)


    A = ssim(img1,img2)
    B = noisy_loss(img2, img1).loss_noisy()
    print("loss_def", B)
    print("ssim_norm",A)
    #test part(end)



