'''
find_pics1.4_alpha
在find_pics1.4基础上制作图形界面
使用UC-Merced数据集
2020/8/28
'''

# 导入界面库
import tkinter as tk
from tkinter import ttk
from tkinter import Menu
from tkinter import filedialog
from tkinter import scrolledtext

# 导入深度学习相关库
import os
import torch
import numpy as np
import torch.nn as nn
from PIL import Image, ImageTk
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms
from resnet50power import ResNet
from resnet50power import Bottleneck as block

# 定义超参数
data_num = 420  # 数据集图像数量
output_num = 10  # 默认输出图片数量
resnet50_path = './model/resnet50_last.pth'  # 网络模型路径
features_path = './feature_lib.npy'  # 特征库路径
language = True  # 默认使用英文界面
data_set_path = './data/test/'  # 数据集路径
feature_library = './feature_lib.npy'          # 特征库
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 使用GPU加速运行程序

# 定义全局参数
input_img = None  # 窗口显示输入图像
label_img = []  # 输出图像

# 界面文字提示语句
str_title = 'Find Pics 1.4 Alpha'
str_operate = 'Operate'
str_image = 'Image'
str_input_image = ' Input Image '
str_input_image_path = 'Input image path:'
str_Browse = 'Browse'
str_Find = 'Find'
str_rate_1 = "Rate(1/5):Receive picture completed\n"
str_rate_2 = "Rate(2/5):Extraction feature completed\n"
str_rate_3 = "Rate(3/5):Calculation similarity completed\n"
str_rate_4 = "Rate(4/5):Get similar images path\n"
str_rate_5 = "Rate(5/5):Show similar images\n"
str_Time = 'Time '
str_output_nums = 'Output numbers:'
str_language = "Chinese"
str_Exit = "Exit"
str_File = "File"
str_About = "About"
str_Help = "Help"
str_output_title = 'Output Image'


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

# 特征相似度计算
def feature_similarity(img_feature, features):
    simi_list = []
    img_feature = torch.Tensor(img_feature)
    for i in range(data_num):
        feature = features[:, i]
        feature = torch.Tensor(feature).view(2048, 1, 1)
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
    output_indices = sorted_indices[0:output_num]
    return output_indices


# 输出图像界面
class output_window():
    def __init__(self, output_indices):
        global label_img, output_num
        output_indices.sort()
        length = output_num
        count = 0
        i = 0
        label_img = []
        dirs = os.listdir(data_set_path)
        for img_name in dirs:
            if i == output_indices[count][0]:
                count = count + 1
                img_name_full = data_set_path + img_name
                output = Image.open(img_name_full)
                output = output.resize((80, 80))
                output = ImageTk.PhotoImage(image=output)
                label_img.append(output)
            i = i + 1
            if count >= length:
                break
        self.root = tk.Toplevel()
        self.root.title(str_output_title)
        for i in range(int(length/5)):
            for j in range(5):
                temp = tk.Label(self.root, image=label_img[i*5+j], width=80, height=80)
                temp.grid(column=j, row=i, padx=4, pady=2)
        self.root.mainloop()


# 程序窗口
class window():
    def __init__(self):
        self.root = tk.Tk()
        self.root.title(str_title)
        # 两个标签页
        tabControl = ttk.Notebook(self.root)
        tab1 = ttk.Frame(tabControl)
        tabControl.add(tab1, text=str_operate)
        tab2 = ttk.Frame(tabControl)
        tabControl.add(tab2, text=str_image)
        tabControl.pack(expand=1, fill="both")

        # 布置第一个标签页
        mighty1 = ttk.LabelFrame(tab1, text=str_input_image)
        mighty1.grid(column=0, row=0, padx=8, pady=4)
        a_grid = ttk.Label(mighty1, text=str_input_image_path)
        a_grid.grid(column=0, row=0, sticky='W')  # 添加一个标签
        input_image_path = tk.StringVar()
        input_image_path_entered = ttk.Entry(mighty1, width=25, textvariable=input_image_path)
        input_image_path_entered.grid(column=0, row=1, sticky='W')  # 添加一个输入框

        def get_path():
            file_path = filedialog.askopenfilename()
            input_image_path.set(file_path)
        tk.Button(mighty1, text=str_Browse, command=get_path).grid(column=1, row=1, sticky='W')  # 添加一个浏览按钮

        def find_pics():
            global input_img
            # 布置第二个标签页, 用来展示输入图像
            mighty2 = ttk.LabelFrame(tab2, text=str_input_image)
            input_img = Image.open(str(input_image_path.get()))
            input_img = input_img.resize((224, 224))
            input_img = ImageTk.PhotoImage(image=input_img)
            Label_img = tk.Label(mighty2, image=input_img, width=224, height=224)
            Label_img.pack()  # 展示输入图像
            mighty2.grid(column=0, row=0)
            # 下面是检索过程
            img_path = str(input_image_path.get())
            scr.insert(tk.END, img_path + "\n")
            scr.insert(tk.END, str_rate_1)  # 滑条展示信息插入
            img_feature = get_img_feature(img_path, model)  # 获取输入图像中层特征
            scr.insert(tk.END, str_rate_2)
            simi_list = feature_similarity(img_feature, features)  # 与特征库进行相似度计算
            scr.insert(tk.END, str_rate_3)
            output_indices = similarity_sort(simi_list)  # 获取输出图像位置
            scr.insert(tk.END, str_rate_4)
            scr.insert(tk.END, str_rate_5)

            def output_img(output_indices):
                # 将相似图像展示到另一个窗口
                output_win = output_window(output_indices)
                output_win.root.mainloop()
            output_img(output_indices)
        find_button = ttk.Button(mighty1, text=str_Find, width=4, command=find_pics)
        find_button.grid(column=1, row=3)  # 添加一个检索按钮

        ttk.Label(mighty1, text=str_output_nums).grid(column=0, row=2, sticky='W')
        number = tk.StringVar()
        number_chosen = ttk.Combobox(mighty1, width=10, textvariable=number, state='readonly')
        number_chosen['values'] = (5, 10, 15, 20, 40)
        number_chosen.grid(column=0, row=3, sticky='W')  # 选择输入图片数量
        number_chosen.current(1)  # 默认输出图片数量=10

        def set_output_num(temp):
            global output_num
            output_num = int(number_chosen.get())
        number_chosen.bind("<<ComboboxSelected>>", set_output_num)

        scr = scrolledtext.ScrolledText(mighty1, width=30, height=10, wrap=tk.WORD)
        scr.grid(column=0, row=5, sticky='WE', columnspan=3)  # 滑条展示信息

        menu_bar = Menu(self.root)
        self.root.config(menu=menu_bar)  # 添加菜单
        file_menu = Menu(menu_bar, tearoff=0)

        def change_language():
            global str_title, str_operate, str_image, str_input_image, str_input_image_path
            global str_Browse, str_Find, str_Time, str_output_nums, str_language, str_Exit
            global str_rate_1, str_rate_2, str_rate_3, str_rate_4, str_rate_5
            global language, str_File, str_About, str_Help, str_output_title
            if language:
                str_title = '检索软件1.4 Alpha'
                str_operate = '操作'
                str_image = '图像'
                str_input_image = ' 输入图像 '
                str_input_image_path = '图片路径:'
                str_Browse = '浏览'
                str_Find = '检索'
                str_rate_1 = "进度(1/5):接收图像成功\n"
                str_rate_2 = "进度(2/5):提取图像中层特征完成\n"
                str_rate_3 = "进度(3/5):相似度计算完成\n"
                str_rate_4 = "进度(4/5):获取相似图像路径成功\n"
                str_rate_5 = "进度(5/5):展示相似图像\n"
                str_Time = '时长 '
                str_output_nums = '输出图像数量:'
                str_language = "英文"
                str_Exit = "退出"
                str_File = "文件"
                str_About = "关于"
                str_Help = "帮助"
                str_output_title = '输出图像'
                language = False
            else:
                str_title = 'Find Pics 1.4 Alpha'
                str_operate = 'Operate'
                str_image = 'Image'
                str_input_image = ' Input Image '
                str_input_image_path = 'Input image path:'
                str_Browse = 'Browse'
                str_Find = 'Find'
                str_rate_1 = "Rate(1/5):Receive picture completed\n"
                str_rate_2 = "Rate(2/5):Extraction feature completed\n"
                str_rate_3 = "Rate(3/5):Calculation similarity completed\n"
                str_rate_4 = "Rate(4/5):Get similar images path\n"
                str_rate_5 = "Rate(5/5):Show similar images\n"
                str_Time = 'Time '
                str_output_nums = 'Output numbers:'
                str_language = "Chinese"
                str_Exit = "Exit"
                str_File = "File"
                str_About = "About"
                str_Help = "Help"
                str_output_title = 'Output Image'
                language = True
            self.refresh()
        file_menu.add_command(label=str_language, command=change_language)
        file_menu.add_separator()

        def quit_():
            self.root.quit()
            self.root.destroy()
            exit()
        file_menu.add_command(label=str_Exit, command=quit_)
        menu_bar.add_cascade(label=str_File, menu=file_menu)
        help_menu = Menu(menu_bar, tearoff=0)
        help_menu.add_command(label=str_About)
        menu_bar.add_cascade(label=str_Help, menu=help_menu)

        input_image_path_entered.focus()
        self.root.mainloop()

    # 中英文切换时刷新窗口
    def refresh(self):
        global win
        self.root.destroy()
        win = window()




# 程序运行入口
if __name__ == '__main__':
    # 加载模型
    # resnet50 = models.resnet50(pretrained=False)
    # resnet50.load_state_dict(torch.load(resnet50_path))
    # model = nn.Sequential(*list(resnet50.children())[:-1])  # 定位到
    # model.eval()

    # 加载模型
    # Resnet50 + CBAM
    model_path = './model/resnet50_last.pth'
    resnet50 = ResNet(block, [3, 4, 6, 3]).cuda()
    resnet50.load_state_dict(torch.load(model_path))
    extract_list = ["conv1", "bn1", "relu", "ca", "sa", "maxpool", "layer1", "layer2", "layer3", "layer4", "ca1", "sa1"]
    resnet50 = FeatureExtractor_model(resnet50, extract_list)
    model = resnet50
    model.eval()

    features = np.load(feature_library)

    # 实例化一个窗口
    win = window()
    win.root.mainloop()
