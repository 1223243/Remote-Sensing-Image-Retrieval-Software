import time
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from getdata import UCDataset as UCD
from torch.utils.data import DataLoader as DataLoader
# from network import Net
import torch
from torch.autograd import Variable
import torch.nn as nn
from matplotlib import pyplot as plt
from torchvision import models
from torchsummary import summary
from resnet50power import ResNet
from resnet50power import Bottleneck as block
# loss下降曲线的绘制
def plot_curve(data):
    fig = plt.figure()
    plt.plot(range(len(data)), data, color='blue')
    plt.legend(['value'], loc='upper right')
    plt.xlabel('step')
    plt.ylabel('value')
    plt.show()

dataset_dir = 'UC_Merced/train/'   # 数据集路径
model_cp = 'model/'    # 网络参数保存位置
# model_file = 'D:/pythonCNN程序/UC-Merced项目/vgg16.pth'

# dataset_dir ='data/'
# model_cp ='/mnt/UC-Merced项目/model/'
# model_file ='vgg16.pth'
model_file ='resnet50.pth'
# model_file ='inception_v3.pth'
# model_file ='densenet169.pth'

workers = 10                 # PyTorch读取数据线程数量
batch_size = 30              # batch_size大小
lr = 0.001                  # 学习率
nepoch =100                   # 训练次数


# 中间层特征提取
class FeatureExtractor_model(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor_model, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    # 自己修改forward函数
    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            print("=======================开始========================")
            print("进入之前大小为")
            print(x.shape)
            print(name)    #打印模型每一层的名字

            if name is "fc":
                x = x.view(x.size(0), -1)
            # if name is "ca" or name is "sa" or name is "ca1" or name is "sa1":#需不需要对sa1和ca1特判
            elif (name is "ca") or (name is "sa") or (name is "ca1") or (name is "sa1"):#需要特判一下，这个注意力机制，比较恶心是model(x)*x
                x = module(x)*x
            else:
                x=module(x)

            print(module)
            print("进入之后变为")
            print(x.shape)
            if name in self.extracted_layers:   #只将需要的特征层extracted_layers输出
                outputs.append(x)
                print("-----------------------------------------------")
                print("该层是需要的特征层，大 小为")
                print(x.shape)
            print("***********************结束************************")
        return outputs



def attention():
    # model = models.resnet50(pretrained=False)
    # model.load_state_dict(torch.load(model_file))  # 加载训练好的模型参数
    # # model = model.cuda()
    #
    # frozen_layers = [model.conv1, model.bn1, model.relu, model.maxpool,
    #                  model.layer1, model.layer2, model.layer3]
    # for layer in frozen_layers:
    #     for name, value in layer.named_parameters():
    #         value.requires_grad = False
    #
    # model.fc.out_features = 21

    # 自己重写的网络
    net = ResNet(block,[3, 4, 6, 3])
    # 需要加载的预训练参数
    resnet = models.resnet50(pretrained=True)
    # 在重写的网络中加载预训练网络
    new_state_dict = resnet.state_dict()
    dd = net.state_dict()
    for k in new_state_dict.keys():
        # print(k)
        if k in dd.keys() and not k.startswith('fc'):  # 不加载全连接层
            # print('yes')
            dd[k] = new_state_dict[k]
    net.load_state_dict(dd)
    net.fc.out_features=21
    # print(net)


    # print(net)


    # extract_list = nn.Sequential(*list(net.children())[:-4])
    # net=FeatureExtractor(net, extract_list)


    # net = models.resnet50(pretrained=False)
    # extract_list = nn.Sequential(*list(net.children())[:-2])
    # extract_list = ["conv1","bn1","relu","maxpool","layer1","layer2","layer3","layer4"]
    # print("11111111111")
    # print(extract_list)
    # print("1111111111")
    # net = FeatureExtractor(net, extract_list)
    # print(net)

    # net = ResNet(block, [3, 4, 6, 3])
    #下面的extract_list需要和FeatureExtractor里面forward 的特判if探讨一下
    # extract_list = ["conv1", "bn1", "relu", "ca", "sa", "maxpool", "layer1", "layer2", "layer3", "layer4"]
    net.cuda()
    extract_list=["conv1","bn1","relu","ca","sa","maxpool","layer1","layer2","layer3","layer4","ca1","sa1"]
    net = FeatureExtractor_model(net, extract_list)

    # 使用summary(model, (3, 224, 224)) 需要加载torchsummary
    # pip install torchsummary
    summary(net, (3,224,224))


if __name__=='__main__':
    attention()
    # net = models.resnet50(pretrained=False)
    # print(net)
    # 问题描述：测试发现 注意力机制把x.shape变小了
