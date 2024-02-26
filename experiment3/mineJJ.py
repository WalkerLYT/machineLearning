import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision
from utils import plot_image, plot_curve, one_hot

batch_size = 256
# 加载数据集
dataset = torchvision.datasets.MNIST('./data/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))    # 使得数据在0附近均匀分布
                               ]))
# 只取10%的样本
train_sampler = SubsetRandomSampler(np.random.choice(len(dataset), int(0.1 * len(dataset)), replace=False))

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
test_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


# 建立神经网络模型
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # 创建卷积层（5*5的卷积核，28-(5-1)=24），灰度图像的输入通道数为1
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # 创建线性层（即全连接层）
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # 2*2窗口池化原特征图
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        # 将(1, 20, 4, 4) 打平成320
        x = x.view(-1, 320)

        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)

        return x

net = Net()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

train_loss = []

# 开始训练
for epoch in range(10):
    for batch_idx, (x, y) in enumerate(train_loader):
        out = net(x)
        y_onehot = one_hot(y)
        loss = F.mse_loss(out, y_onehot)

        # 计算梯度，以便反向传播修改参数（三件套↓）
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

        if batch_idx % 10 == 0:
            print(epoch, batch_idx, loss.item())

plot_curve(train_loss)  # 绘制曲线


# 测试
total_correct = 0
for x, y in test_loader:
    out = net(x)
    predict = out.argmax(dim=1)
    correct = predict.eq(y).sum().float().item()
    total_correct += correct

total_num = len(test_loader.dataset)
acc = total_correct / total_num
print('准确率为：', acc)

# 随机选一个batch（test_loader设置了shuffle为True），绘制其具体的测试结果图，看看机器队是否打败了人工队
x, y = next(iter(test_loader))
out = net(x)
predict = out.argmax(dim=1)  # 返回最大值对应的标签
plot_image(x, predict, 'test_result')

total_correct = 0
for x, y in test_loader:
    out = net(x)
    predict = out.argmax(dim=1)
    correct = predict.eq(y).sum().float().item()
    total_correct += correct

total_num = len(test_loader.dataset)
acc = total_correct / total_num
print('准确率为：', acc)














