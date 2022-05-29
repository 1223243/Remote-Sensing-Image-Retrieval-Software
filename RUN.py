
import os
import time
import torch
import shutil
import numpy as np
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms
from torchsummary import summary
from resnet50power import ResNet
from resnet50power import Bottleneck as block

# 定义全局变量
output_img_num = 20                       # 输出output_img_num幅相似图像
data_set_path = 'UC-Merced/data/test/'              # 测试集路径
data_num = 420                           # 测试集图片数量
feature_library = 'feature_lib.npy'          # 特征库


# 得到输入图像的中层特征
def get_img_feature(imgpath, model):
    img_to_tensor = transforms.ToTensor()

    # 提取中层特征函数
    def extract_feature(model, imgpath):
        img = Image.open(imgpath)
        img = img.resize((224, 224))
        img = img.convert("RGB")
        tensor = img_to_tensor(img).cuda()
        tensor = tensor.resize_(1, 3, 224, 224)
        result = model(Variable(tensor))
        result = F.max_pool2d(result, kernel_size=7, stride=7)
        result_npy = result.data.cpu().numpy()
        return result_npy[0]
    feature = extract_feature(model, imgpath)
    return feature


# 特征相似度计算
def feature_similarity(img_feature, features):
    simi_list = []
    img_feature = torch.Tensor(img_feature)
    for i in range(data_num):
        feature = features[:, i]
        feature = torch.Tensor(feature).view(2048, 1, 1)
        print(img_feature, feature)
        simi = torch.cosine_similarity(img_feature, feature, dim=0)
        simi = simi.numpy()  # tensor转numpy
        # print(simi)
        simi = simi.tolist()
        simi_list.append(simi)
    return simi_list


# 特征相似度排序
def similarity_sort(simi_list):
    sorted_indices = []
    sorted_indices = sorted(enumerate(simi_list), key=lambda x:x[1])
    sorted_indices.reverse()
    output_indices = sorted_indices[0:output_img_num]
    return output_indices


# 复制相似图像到文件夹中
def output_img(output_indices):
    output_indices.sort()
    length = output_img_num
    count = 0
    i = 0
    dirs = os.listdir(data_set_path)
    for img_name in dirs:
        if i == output_indices[count][0]:
            print(img_name, output_indices[count][1])  # 显示图像相似度
            count = count+1
            img_name_full = data_set_path+img_name
            new_folder = 'output_image'
            shutil.copy(img_name_full, new_folder)
        i = i+1
        if count >= length:
            break


# 中间层特征提取
class FeatureExtractor_model(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor_model, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    # 自己修改forward函数
    def forward(self, x):
        outputs = None
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
                if flag:
                    outputs = x
                else:
                    torch.cat([outputs, x], dim=1)
        return outputs


if __name__ == '__main__':
    model_path = './model/resnet50_last.pth'
    # 加载模型
    resnet50 = ResNet(block, [3, 4, 6, 3]).cuda()
    # summary(resnet50, (3, 224, 224))
    resnet50.load_state_dict(torch.load(model_path))
    # model = nn.Sequential(*list(resnet50.children())[:-1])  # 定位到
    extract_list = ["conv1", "bn1", "relu", "ca", "sa", "maxpool", "layer1", "layer2", "layer3", "layer4", "ca1", "sa1"]
    resnet50 = FeatureExtractor_model(resnet50, extract_list)
    model = resnet50
    model.eval()
    
    features = np.load(feature_library)

    img_name = input("请输入待检索图像的完整名字:\n")
    since = time.time()                   # 计时开始
    img_path = 'input_image/'+img_name                     # 输入图像
    print("进度:1/5.接收图像成功")
    img_feature = get_img_feature(img_path, model)         # 获取输入图像中层特征

    # maxind = 0
    # num = img_feature.tolist()
    # for i in range(21):
    #     if num[i] > num[maxind]:
    #         maxind = i
    # print(maxind)

    print("进度:2/5.提取图像中层特征完成")
    simi_list = feature_similarity(img_feature, features)  # 与特征库进行相似度计算
    print("进度:3/5.相似度计算完成")
    output_indices = similarity_sort(simi_list)            # 获取输出图像位置
    print("进度:4/5.成功获取"+str(output_img_num)+"幅相似图像")
    output_img(output_indices)                             # 输出图像
    print("进度:5/5.图像已复制到\"output_image\"文件夹")
    time_elapsed = time.time() - since    # 计时结束
    print('检索时长 {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))




