'''
"input_image"文件夹中放置一系列待检索图片
改好全局变量
改好模型载入代码
运行程序
计算网络模型的mAP
'''
import os
import time
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms
from resnet50power import ResNet
from resnet50power import Bottleneck as block

#定义全局变量
output_img_num = 420                     #输出相似图像, 必须与数据集图片数量一致
data_set_path = './data/test/'              #数据集路径
data_num = 420                           #数据集图片数量
feature_library = 'feature_lib.npy'   #特征库
input_image = './data/test/'              #输入图片路径
mid_feature = 2048                         #vgg16为512, resnet50为2048


#得到输入图像的中层特征
def get_img_feature(imgpath, model):
    img_to_tensor = transforms.ToTensor()
    #提取中层特征函数
    def extract_feature(model, imgpath):
        img = Image.open(imgpath)
        img = img.resize((224,224))
        tensor = img_to_tensor(img).cuda()
        tensor = tensor.resize_(1,3,224,224)
        result = model(Variable(tensor))
        result = F.max_pool2d(result, kernel_size=7, stride=7)
        result_npy = result.data.cpu().numpy()
        return result_npy[0]
    feature = extract_feature(model, imgpath)
    return feature

#特征相似度计算
def feature_similarity(img_feature, features):
    simi_list = []
    img_feature = torch.Tensor(img_feature)
    # img_feature = F.max_pool2d(img_feature, kernel_size=7, stride=7)
    for i in range(data_num):
        feature = features[:,i]
        feature = torch.Tensor(feature).view(mid_feature,1,1)
        simi = torch.cosine_similarity(img_feature, feature, dim=0)
        simi = simi.numpy() #tensor转numpy
        simi = simi.tolist()
        simi_list.append(simi)
    return simi_list

#特征相似度排序
def similarity_sort(simi_list):
    sorted_indices = []
    sorted_indices = sorted(enumerate(simi_list), key=lambda x:x[1])
    sorted_indices.reverse()
    output_img_indices = sorted_indices[0:output_img_num]
    return output_img_indices


def get_img_belong_file(img_name):
    temp = str(img_name)[:-6]
    # print(temp+"  -")
    img_belong_file = temp
    # print(img_belong_file+"  =")
    return img_belong_file

def convert_to_sequence_and_index(output_img_indices):
    index_sequence = [[] for i in range(output_img_num)]
    for i in range(output_img_num):
        temp = [output_img_indices[i][0], i]
        index_sequence[i] = temp
    return index_sequence
    
def get_output_img_name_sequence(sorted_index_sequence):
    name_sequence = [[] for i in range(output_img_num)]
    dirs = os.listdir(data_set_path)
    index = 0
    count = 0
    for img_name in dirs:
        if index == sorted_index_sequence[count][0]:
            temp = [img_name, sorted_index_sequence[count][1]]
            name_sequence[count] = temp
            count += 1
        index += 1
        if count >= output_img_num:
            break
    return name_sequence
            
def get_ordered_output_file_names(name_sequence):
    ordered_output_file_names = []
    name_sequence = sorted(name_sequence,key=(lambda x:x[1]))
    for i in range(output_img_num):
        img_belong_file = get_img_belong_file(name_sequence[i][0])
        ordered_output_file_names.append(img_belong_file)
    return ordered_output_file_names

def get_AP(img_belong_file, ordered_output_file_names):
    AP = 0
    count = 0
    for i in range(output_img_num):
        # print(img_belong_file,ordered_output_file_names[i])
        if img_belong_file == ordered_output_file_names[i]:
            count += 1
            AP += count/(i+1)
    AP = AP / count
    return AP


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
    # Resnet50 + CBAM
    resnet50 = ResNet(block, [3, 4, 6, 3]).cuda()
    # summary(resnet50, (3, 224, 224))
    resnet50.load_state_dict(torch.load(model_path))
    # model = nn.Sequential(*list(resnet50.children())[:-1])  # 定位到
    extract_list = ["conv1", "bn1", "relu", "ca", "sa", "maxpool", "layer1", "layer2", "layer3", "layer4", "ca1", "sa1"]
    resnet50 = FeatureExtractor_model(resnet50, extract_list)
    model = resnet50

    # # Resnet50 预训练的
    # model = models.resnet50(pretrained=False)
    # model_file = './resnet50.pth'
    # model.load_state_dict(torch.load(model_file))  # 加载训练好的模型参数
    # model = nn.Sequential(*list(model.children())[:-2])
    # model.cuda()


    model.eval()
    features = np.load(feature_library)
    
    since = time.time()  # 计时开始

    mAP = 0
    cnt = 0
    dirs = os.listdir(input_image)
    for img_name in dirs:
        img_belong_file = get_img_belong_file(img_name)#获取图片所属类别
        #开始检索
        img_path = input_image+img_name                    #输入图像
        img_feature = get_img_feature(img_path, model)        #获取输入图像中层特征
        simi_list = feature_similarity(img_feature, features) #与特征库进行相似度计算
        output_img_indices = similarity_sort(simi_list)       #获取输出图像位置
        #后面是一系列对输出图片文件名的处理
        #为了得到有序的输出图片的所属文件名
        index_sequence = convert_to_sequence_and_index(output_img_indices)
        sorted_index_sequence = sorted(index_sequence,key=(lambda x:x[0]))
        name_sequence = get_output_img_name_sequence(sorted_index_sequence)
        ordered_output_file_names = get_ordered_output_file_names(name_sequence)
        # print(ordered_output_file_names)
        #计算AP查询检索精度
        AP = get_AP(img_belong_file, ordered_output_file_names)
        mAP += AP
        cnt += 1
        print("第", cnt, "幅图片的AP:", AP)
    mAP = mAP / cnt
    print("\n模型的mAP为:", mAP)
    
    time_elapsed = time.time() - since    #计时结束
    print('检索时长 {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
