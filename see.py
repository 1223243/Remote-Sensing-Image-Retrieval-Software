import time


import getdata
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from matplotlib import pyplot as plt
from torchvision import models, transforms
from torchsummary import summary
from resnet50power import ResNet
from resnet50power import Bottleneck as block
from PIL import Image
import torch.nn.functional as F


# 提取特征
def extract_feature(model, imgpath):
    # print(model)
    img = Image.open(imgpath)
    img = img.resize((224, 224))
    tensor = dataTransform(img).cuda()
    tensor = tensor.resize_(1, 3, 224, 224)
    result = model(Variable(tensor))
    # print(torch.Tensor(result).shape)
    # result = torch.tensor([item.cpu().detach().numpy() for item in result]).cuda()
    # result = F.max_pool2d(result, kernel_size=7, stride=7)
    result_npy = result.data.cpu().numpy()
    return result_npy[0]

# 中间层特征提取
class FeatureExtractor_model(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor_model, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    # 自己修改forward函数
    def forward(self, x):
        outputs = None
        # outputs=[]
        flag = True
        for name, module in self.submodule._modules.items():
            if name is "fc":
                x = x.view(x.size(0), -1)
            # if name is "ca" or name is "sa" or name is "ca1" or name is "sa1":#需不需要对sa1和ca1特判
            elif (name is "ca") or (name is "sa") or (name is "ca1") or (name is "sa1"):#需要特判一下，这个注意力机制，比较恶心是model(x)*x
                x = module(x)*x
            else:
                x=module(x)
            if name in self.extracted_layers:   #只将需要的特征层extracted_layers输出
                # outputs.append(x)
                if flag:
                    outputs = x
                else:
                    torch.cat([outputs, x], dim=1)

        # print(outputs.shape)
        return outputs






if __name__ == '__main__':
    model_path = './model/resnet50_last.pth'
    # 默认输入网络的图片大小
    IMAGE_SIZE = 224

    # 定义一个转换关系，用于将图像数据转换成PyTorch的Tensor形式
    dataTransform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),  # 将图像按比例缩放至合适尺寸
        transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),  # 从图像中心裁剪合适大小的图像
        transforms.ToTensor()  # 转换成Tensor形式，并且数值归一化到[0.0, 1.0]，同时将H×W×C的数据转置成C×H×W，这一点很关键
    ])

    # Resnet50
    resnet50 = ResNet(block, [3, 4, 6, 3]).cuda()
    resnet50.load_state_dict(torch.load(model_path))
    # model = nn.Sequential(*list(resnet50.children())[:-1]).cuda()  # 定位到
    extract_list = ["conv1", "bn1", "relu", "ca", "sa", "maxpool", "layer1", "layer2", "layer3", "layer4", "ca1", "sa1"]
    resnet50 = FeatureExtractor_model(resnet50, extract_list)
    # summary(resnet50, (3, 224, 224))
    model = resnet50
    model.eval()

    # 数据集路径
    imgpath = './data/test/agricultural81.tif'
    feature=extract_feature(model, imgpath)
    print(feature.shape)
    for feature_map in feature:
        # print(feature_map.shape)
        # # [N, C, H, W] -> [C, H, W]
        # im = feature_map.numpy()
        # [C, H, W] -> [H, W, C]
        # im = np.transpose(feature_map, [1, 2, 0])

        # show top 12 feature maps
        plt.figure()
        for i in range(7):
            ax = plt.subplot(3, 3, i + 1)
            # [H, W, C]

            cmap = 'nipy_spectral'
            plt.imshow(feature_map[:,:], cmap=plt.get_cmap(cmap))
        plt.show()


















