import re

from torchvision import models
from getdata import UCDataset as UCD
# from network import Net
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import random
import os

import getdata
from getdata import UCDataset as UCD
from resnet50power import ResNet as MyResNet
from resnet50power import Bottleneck as block
# dataset_dir = 'D:/pythonCNN程序/UC-Merced项目/data/test/'                    # 数据集路径
# model_file = 'D:/pythonCNN程序/UC-Merced项目/vgg16.pth'

dataset_dir = 'data/test/'
model_file = 'D:/Python_code/UC-Merced/model/resnet50_last.pth'

# 模型保存路径
N = 100                                         #一下预测 N 张

def test():

    # setting model
    # model = Net()                                       # 实例化一个网络
    # ------------------GPU---------------------
    # model.cuda()                                        # 送入GPU，利用GPU计算
    # model = nn.DataParallel(model)
    # model=models.resnet50(pretrained=False)
    # model = models.inception_v3(pretrained=False)
    model=MyResNet(block,[3, 4, 6, 3])
    model.load_state_dict(torch.load(model_file))       # 加载训练好的模型参数
    model = model                               #GPU
    model.eval()                                        # 设定为评估模式，即计算过程中不要dropout

    # get data
    files = random.sample(os.listdir(dataset_dir), N)   # 随机获取N个测试图像
    imgs = []           # img
    imgs_data = []      # img data
    name_segment = []   #namesegment
    for file in files:
        # print(file)            #打印图片路径名
        namesegment = re.findall('([a-zA-Z]+|\d+)', file)
        # print(namesegment[0])  # 图像的类别标记号
        # print(namesegment[1])  # 图像的类别名

        img = Image.open(dataset_dir + file)            # 打开图像
        img_data = getdata.dataTransform(img)           # 转换成torch tensor数据
        # img_data = Variable(img_data)
        img_data= Variable(img_data)                 #GPU

        name_segment.append(str(namesegment[1]))             # 图像的类别名
        imgs.append(img)                                # 图像list
        imgs_data.append(img_data)                      # tensor list
    imgs_data = torch.stack(imgs_data)                  # tensor list合成一个4D tensor


    if hasattr(torch.cuda, 'empty_cache'):  # 释放GPU
        torch.cuda.empty_cache()
    out = model(imgs_data)  # 对每个图像进行网络计算
    out = F.softmax(out, dim=1)  # 输出概率化
    out = out.data.cpu().numpy()  # 转成numpy数据

    accnum=0                                               #正确数
    # pring results         显示结果
    for idx in range(N):
        # calculation
        plt.figure()
        maxnum=-1
        j=-1
        for i in range(21):
            # print('{0}: {1}'.format(i,out[idx,i]))
            if out[idx,i]>maxnum:
                maxnum=out[idx,i]
                j=i
        # print('第{0}张max类别编号:{1}'.format(idx+1,j))     #打印概率最大的类别编号
        # if(j*20<=int(name_segment[idx]) and int(name_segment[idx])<=j*20+19): accnum+=1
        if 0 ==j:
            if (name_segment[idx] == 'agricultural'):
                accnum += 1
                # print(0)
            else:
                print('第{0}张正确类别是:{1}'.format(idx + 1, str(name_segment[idx])))
                print('第{0}张测试类别是:agricultural'.format(idx + 1 ))
            plt.suptitle('agricultural:{:.1%}'.format(maxnum))#'农田'
        elif 1 ==j:
            if (name_segment[idx] == 'airplane'):
                accnum += 1
                # print(1)
            else:
                print('第{0}张正确类别是:{1}'.format(idx + 1, str(name_segment[idx])))
                print('第{0}张测试类别是:airplane'.format(idx + 1))
            plt.suptitle('airplane:{:.1%}'.format(maxnum))
        elif 2 ==j:
            if (name_segment[idx] == 'baseballdiamond'):
                accnum += 1
                # print(2)
            else:
                print('第{0}张正确类别是:{1}'.format(idx + 1, str(name_segment[idx])))
                print('第{0}张测试类别是:baseballdiamond'.format(idx + 1))
            plt.suptitle('baseballdiamond:{:.1%}'.format(maxnum))#棒球场
        elif 3 ==j:
            if (name_segment[idx] == 'beach'):
                accnum += 1
                # print(3)
            else:
                print('第{0}张正确类别是:{1}'.format(idx + 1, str(name_segment[idx])))
                print('第{0}张测试类别是:beach'.format(idx + 1))
            plt.suptitle('beach:{:.1%}'.format(maxnum))
        elif 4 ==j:
            if (name_segment[idx] == 'buildings'):
                accnum += 1
                # print(4)
            else:
                print('第{0}张正确类别是:{1}'.format(idx + 1, str(name_segment[idx])))
                print('第{0}张测试类别是:buildings'.format(idx + 1))
            plt.suptitle('buildings:{:.1%}'.format(maxnum))
        elif 5 ==j:
            if (name_segment[idx] == 'chaparral'):
                accnum += 1
                # print(5)
            else:
                print('第{0}张正确类别是:{1}'.format(idx + 1, str(name_segment[idx])))
                print('第{0}张测试类别是:chaparral'.format(idx + 1))
            plt.suptitle('chaparral{:.1%}'.format(maxnum))#阔叶灌丛
        elif 6 ==j:
            if (name_segment[idx] == 'denseresidential'):
                accnum += 1
                # print(6)
            else:
                print('第{0}张正确类别是:{1}'.format(idx + 1, str(name_segment[idx])))
                print('第{0}张测试类别是:denseresidential'.format(idx + 1))
            plt.suptitle('denseresidential:{:.1%}'.format(maxnum))#密集的
        elif 7 ==j:
            if (name_segment[idx] == 'forest'):
                accnum += 1
                # print(7)
            else:
                print('第{0}张正确类别是:{1}'.format(idx + 1, str(name_segment[idx])))
                print('第{0}张测试类别是:forest'.format(idx + 1))
            plt.suptitle('forest:{:.1%}'.format(maxnum))
        elif 8 ==j:
            if (name_segment[idx] == 'freeway'):
                accnum += 1
                # print(8)
            else:
                print('第{0}张正确类别是:{1}'.format(idx + 1, str(name_segment[idx])))
                print('第{0}张测试类别是:freeway'.format(idx + 1))
            plt.suptitle('freeway:{:.1%}'.format(maxnum))#高速公路
        elif 9 ==j:
            if (name_segment[idx] == 'golfcourse'):
                accnum += 1
                # print(9)
            else:
                print('第{0}张正确类别是:{1}'.format(idx + 1, str(name_segment[idx])))
                print('第{0}张测试类别是:golfcourse'.format(idx + 1))
            plt.suptitle('golfcourse:{:.1%}'.format(maxnum))#高尔夫球场
        elif 10 ==j:
            if (name_segment[idx] == 'harbor'):
                accnum += 1
                # print(10)
            else:
                print('第{0}张正确类别是:{1}'.format(idx + 1, str(name_segment[idx])))
                print('第{0}张测试类别是:harbor'.format(idx + 1))
            plt.suptitle('harbor:{:.1%}'.format(maxnum))#港口
        elif 11 ==j:
            if (name_segment[idx] == 'intersection'):
                accnum += 1
                # print(11)
            else:
                print('第{0}张正确类别是:{1}'.format(idx + 1, str(name_segment[idx])))
                print('第{0}张测试类别是:intersection'.format(idx + 1))
            plt.suptitle('intersection:{:.1%}'.format(maxnum))#交叉口
        elif 12 ==j:
            if (name_segment[idx] == 'mediumresidential'):
                accnum += 1
                # print(12)
            else:
                print('第{0}张正确类别是:{1}'.format(idx + 1, str(name_segment[idx])))
                print('第{0}张测试类别是:mediumresidential'.format(idx + 1))
            plt.suptitle('mediumresidential:{:.1%}'.format(maxnum))#中型住宅
        elif 13 ==j:
            if (name_segment[idx] == 'mobilehomepark'):
                accnum += 1
                # print(13)
            else:
                print('第{0}张正确类别是:{1}'.format(idx + 1, str(name_segment[idx])))
                print('第{0}张测试类别是:mobilehomepark'.format(idx + 1))
            plt.suptitle('mobilehomepark:{:.1%}'.format(maxnum))#家庭公园
        elif 14 ==j:
            if (name_segment[idx] == 'overpass'):
                accnum += 1
                # print(14)
            else:
                print('第{0}张正确类别是:{1}'.format(idx + 1, str(name_segment[idx])))
                print('第{0}张测试类别是:overpass'.format(idx + 1))
            plt.suptitle('overpass:{:.1%}'.format(maxnum))#天桥
        elif 15 ==j:
            if (name_segment[idx] == 'parkinglot'):
                accnum += 1
                # print(15)
            else:
                print('第{0}张正确类别是:{1}'.format(idx + 1, str(name_segment[idx])))
                print('第{0}张测试类别是:parkinglot'.format(idx + 1))
            plt.suptitle('parkinglot:{:.1%}'.format(maxnum))
        elif 16 ==j:
            if (name_segment[idx] == 'river'):
                accnum += 1
                # print(16)
            else:
                print('第{0}张正确类别是:{1}'.format(idx + 1, str(name_segment[idx])))
                print('第{0}张测试类别是:river'.format(idx + 1))
            plt.suptitle('river:{:.1%}'.format(maxnum))
        elif 17 ==j:
            if (name_segment[idx] == 'runway'):
                accnum += 1
                # print(17)
            else:
                print('第{0}张正确类别是:{1}'.format(idx + 1, str(name_segment[idx])))
                print('第{0}张测试类别是:runway'.format(idx + 1))
            plt.suptitle('runway:{:.1%}'.format(maxnum))#跑道
        elif 18 ==j:
            if (name_segment[idx] == 'sparseresidential'):
                accnum += 1
                # print(18)
            else:
                print('第{0}张正确类别是:{1}'.format(idx + 1, str(name_segment[idx])))
                print('第{0}张测试类别是:sparseresidential'.format(idx + 1))
            plt.suptitle('sparseresidential:{:.1%}'.format(maxnum))#稀疏住宅
        elif 19 ==j:
            if (name_segment[idx] == 'storagetanks'):
                accnum += 1
                # print(19)
            else:
                print('第{0}张正确类别是:{1}'.format(idx + 1, str(name_segment[idx])))
                print('第{0}张测试类别是:storagetanks'.format(idx + 1))
            plt.suptitle('storagetanks:{:.1%}'.format(maxnum))#储罐
        elif 20==j:
            if (name_segment[idx] == 'tenniscourt'):
                accnum += 1
                # print(20)
            else:
                print('第{0}张正确类别是:{1}'.format(idx + 1, str(name_segment[idx])))
                print('第{0}张测试类别是:tenniscourt'.format(idx + 1))
            plt.suptitle('tenniscourt:{:.1%}'.format(maxnum))#网球场

        plt.imshow(imgs[idx])
        # plt.show()

    print('正确数:{0}个'.format(accnum))
    print('正确率:{0}%'.format((accnum/N)*100))

if __name__ == '__main__':
    test()