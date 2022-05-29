
import time
import torch.nn as nn
import torch
import torchvision.models as models
from torch.autograd import Variable
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import csv
from PIL import Image
import os
import pandas as pd
import numpy
import torch.nn.functional as F
from torchsummary import summary
from resnet50power import ResNet
from resnet50power import Bottleneck as block

feature_size = 2048            # 模型中层特征大小


# 提取特征
def extract_feature(model, imgpath):
    # print(model)
    img = Image.open(imgpath)
    img = img.resize((224, 224))
    tensor = dataTransform(img).cuda()
    tensor = tensor.resize_(1, 3, 224, 224)
    result = model(Variable(tensor))
    result = F.max_pool2d(result, kernel_size=7, stride=7)
    result_npy = result.data.cpu().numpy()
    return result_npy[0]


# 写入所有特征
def setfeaturedata(model, picturedatapath):
    feature_lib = None
    print('开始写入')
    i = 0
    since = time.time()  # 计算写入需要的总时长
    for imgpath in os.listdir(picturedatapath):
        i += 1
        print(i)
        # 这一句hin重要，是个隐藏文件... .. .
        if imgpath != '.DS_Store':
            # os.path.splitext将文件名拆分为名字和后缀名，可以打印出来看一下
            name = os.path.splitext(imgpath)  # imgpath='agricultural00.tif'
            # 获取拆分后的第一个元素(文件名)也就是“agricultural00”
            img_segment = name[0]
            # filename.append(img_segment)
            # print(name[0])  # agricultural00
        path = picturedatapath + imgpath
        temp = extract_feature(model, path)
        temp = torch.FloatTensor(temp).view(feature_size, 1)
        # temp = F.max_pool2d(temp, kernel_size=7, stride=7)  # -------------写入时池化
        if i == 1:
            feature_lib = temp
            continue
        feature_lib = torch.cat([feature_lib, temp], dim=1)
    time_elapsed = time.time() - since
    print(feature_lib.shape)
    feature_lib = feature_lib.view(feature_size, i)
    np.save("feature_lib.npy", feature_lib)
    print('Setfeaturedata complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


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

    model_path='./model/resnet50_last.pth'
    # 默认输入网络的图片大小
    IMAGE_SIZE = 224

    # 定义一个转换关系，用于将图像数据转换成PyTorch的Tensor形式
    dataTransform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),  # 将图像按比例缩放至合适尺寸
        transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),  # 从图像中心裁剪合适大小的图像
        # transforms.RandomHorizontalFlip(p=0.5),#依据概率p对PIL图片进行水平翻转
        # transforms.RandomVerticalFlip(p=0.5),  # 依据概率p对PIL图片进行垂直翻转
        transforms.ToTensor(),  # 转换成Tensor形式，并且数值归一化到[0.0, 1.0]，同时将H×W×C的数据转置成C×H×W，这一点很关键
    ])

    # Resnet50+CBAM
    resnet50 = ResNet(block, [3, 4, 6, 3]).cuda()
    resnet50.load_state_dict(torch.load(model_path))
    # model = nn.Sequential(*list(resnet50.children())[:-1]).cuda()  # 定位到
    extract_list = ["conv1", "bn1", "relu", "ca", "sa", "maxpool", "layer1", "layer2", "layer3", "layer4", "ca1", "sa1"]
    resnet50 = FeatureExtractor_model(resnet50, extract_list)
    # summary(resnet50, (3, 224, 224))
    model = resnet50

    # # # Resnet50 预训练的
    # model = models.resnet50(pretrained=False)
    # model_file = './resnet50.pth'
    # model.load_state_dict(torch.load(model_file))  # 加载训练好的模型参数
    # model = nn.Sequential(*list(model.children())[:-2])
    # model.cuda()

    model.eval()

    # 数据集路径
    picturedatapath = './data/test/'

    setfeaturedata(model, picturedatapath)
