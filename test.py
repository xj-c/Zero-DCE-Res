from torchvision.models import vgg16
import torchvision
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import imageio
import matplotlib.image as pm

model = vgg16(pretrained=True)

features = []
def hook(moudule,input,output):
    features.append(output.cpu())



def close(self):
    self.hook.remove()

def show_feature_map(feature_map):
    feature_map = feature_map.squeeze(0)
    w_val = feature_map.shape[1]
    h_val = feature_map.shape[2]
    feature_map_sum = torch.sum(feature_map, 0)
    image = feature_map_sum / feature_map.shape[0]
    image = image.cpu().detach().numpy()
    pm.imsave("./data/St/" + "test.jpg", image)
    #feature_map = feature_map.cpu().detach().numpy()
    #for i in range(1,feature_map.shape[0]+1):
    #    plt.subplot(i,w_val,h_val)
    #    pm.imsave("./data/St/"+str(i)+".jpg", feature_map[i-1])



if __name__ == '__main__':
    img = Image.open('./01.jpg')
    img = np.asarray(img)
    img = torch.from_numpy(img).float()
    img = img.permute(2,0,1)
    img = img.unsqueeze(0)


    input = torch.randn(1,3,224,224)

    hook1 = model.features[19].register_forward_hook(hook)

    out = model(img)
    hook1.remove()
    print(features[0].shape)
    #show_feature_map(features[0])