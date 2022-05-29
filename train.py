import time
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from getdata import UCDataset as UCD
from torch.utils.data import DataLoader as DataLoader
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


dataset_dir = './data/'
model_cp = 'model/'
model_file = './resnet50.pth'


workers = 5                 # PyTorch读取数据线程数量
batch_size = 30              # batch_size大小
lr = 0.6 / 1024 * 16                 # 学习率
nepoch = 50                   # 训练次数
sample_interval=20              #采样间隔
step_size=7                     #固定步长衰减
def train():
    since = time.time()
    datafile = UCD('train', dataset_dir)  # 实例化一个数据集
    dataloader = DataLoader(datafile, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)

    print('Dataset loaded! length of train set is {0}'.format(len(datafile)))

    #Resnet50 H(x)-x
    model = models.resnet50(pretrained=False)
    model.load_state_dict(torch.load(model_file))  # 加载训练好的模型参数

    net = ResNet(block, [3, 4, 6, 3])
    new_state_dict = model.state_dict()
    dd = net.state_dict()
    for k in new_state_dict.keys():
        if k in dd.keys() and not k.startswith('fc'):  # 不加载全连接层
            dd[k] = new_state_dict[k]
    net.load_state_dict(dd)
    # net.fc.out_features = 21
    model = net

    # frozen_layers = [model.conv1, model.bn1, model.relu, model.maxpool,
    #                 model.layer1, model.layer2, model.layer3]
    # for layer in frozen_layers:
    #     for name, value in layer.named_parameters():
    #         value.requires_grad = False

    # model.fc.out_features=21
# ----------------------------------------------------------------------------------------------------------------------
    model.train()          # 网络设定为训练模式，有两种模式可选，.train()和.eval()，训练模式和评估模式，区别就是训练模式采用了dropout策略，可以放置网络过拟合
    model = model.cuda()
    criterion=nn.CrossEntropyLoss()                          # 定义loss计算方法，cross entropy，交叉熵，可以理解为两者数值越接近其值越小
    criterion.cuda()
    #实例化一个优化器，即调整网络参数(只调整分类层)，优化方式为adam方法
    # Resnet50
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer=torch.optim.SGD(model.parameters(),lr=lr, momentum=0.9)

    # scheduler = MultiStepLR(optimizer,milestones=[30,80],gamma=0.1)   #学习率衰减策略
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    # scheduler= ReduceLROnPlateau(optimizer,'min')
    train_loss = []     # 将loss 保存起来
    for epoch in range(nepoch):
        # 读取数据集中数据进行训练，因为dataloader的batch_size设置为16，所以每次读取的数据量为16，即img包含了16个图像，label有16个
        cnt = 0  # 训练图片数量
        for img,label in dataloader:
            # img, label = Variable(img), Variable(label)
            #------------------GPU-----------------------------------------------
            img,label=Variable(img).cuda(),Variable(label).cuda()
            out=model(img)                      # 计算网络输出值，就是输入网络一个图像数据，输出猫和狗的概率，调用了网络中的forward()方法
            loss=criterion(out,label.squeeze()) # 计算损失，也就是网络输出值和实际label的差异，显然差异越小说明网络拟合效果越好，此处需要注意的是第二个参数，必须是一个1维Tensor
            loss.backward()                     # 误差反向传播，采用求导的方式，计算网络中每个节点参数的梯度，显然梯度越大说明参数设置不合理，需要调整
            optimizer.step()                    # 优化采用设定的优化方法对网络中的各个参数进行调整
            optimizer.zero_grad()               # 清除优化器中的梯度以便下一次计算，因为优化器默认会保留，不清除的话，每次计算梯度都回累加
            cnt+=1                              # 训练图片数量+1
            train_loss.append(loss.item())      # 存储loss值
            print('Epoch:{0},Frame:{1}, train_loss {2}'.format(epoch, cnt * batch_size, loss / batch_size))
            # if cnt==1:
            #     break
        scheduler.step()  # 学习率衰减
        if epoch % sample_interval == 0:
            torch.save(model.state_dict(), '{0}/resnet50_{1}.pth'.format(model_cp, epoch))  # 训练所有数据后，保存网络的参数
    torch.save(model.state_dict(), '{0}/resnet50_last.pth'.format(model_cp))

    # 用图像打印出loss 的改变
    plot_curve(train_loss)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

if __name__=='__main__':
    train()


















