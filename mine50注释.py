import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision
from utils import plot_image, plot_curve, one_hot

batch_size = 256
# 第一步：加载数据集
dataset = torchvision.datasets.MNIST('./data/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))    # 使得数据在0附近均匀分布
                               ]))
# 只取50%的样本
train_sampler = SubsetRandomSampler(np.random.choice(len(dataset), int(0.5 * len(dataset)), replace=False))

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
test_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# 第二步：建立神经网络模型
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # 创建线性层（即全连接层）
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # x: [b, 1, 28,28]
        # h1 = relu(xw1+b1)
        x = F.relu(self.fc1(x))
        # h2 = relu(h1w2+b2)
        x = F.relu(self.fc2(x))
        # h3 = h2w3+b3
        x = F.log_softmax(self.fc3(x), dim=1)

        return x

net = Net()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)     # 返回一个[w1, b1, w2, b2, w3, b3]，lr步长，momentum动量

train_loss = []

# 第三步：开始训练
for epoch in range(3):  # 循环整个数据集三次
    for batch_idx, (x, y) in enumerate(train_loader):
        # view变形函数，[b, 1, 28, 28] -> [b, 784] 整张图片打平成一维向量
        x = x.view(x.size(0), 28*28)
        # -> [b, 10]
        out = net(x)
        y_onehot = one_hot(y)
        # loss为out和y的误差
        loss = F.mse_loss(out, y_onehot)

        # 计算梯度，以便反向传播修改参数（三件套↓）
        optimizer.zero_grad()   # 清理之前batch的梯度，防止发生累加出错
        # PyTorch会自动计算损失函数对每个参数的梯度，并将这些梯度值累加到每个参数对应的grad属性中。接着就能利用这些梯度值来更新神经网络的参数，使神经网络能够更好地拟合训练数据
        loss.backward()
        # w' = w - lr*grad
        optimizer.step()

        train_loss.append(loss.item())

        if batch_idx % 10 == 0:
            print(epoch, batch_idx, loss.item())

plot_curve(train_loss)  # 绘制误差下降曲线


# 第四步：测试
total_correct = 0   # 预测值和实际值匹配正确的数量
for x, y in test_loader:
    x = x.view(x.size(0), 28*28)
    out = net(x)
    predict = out.argmax(dim=1)     # 返回最大值对应的标签
    correct = predict.eq(y).sum().float().item()
    total_correct += correct

total_num = len(test_loader.dataset)
acc = total_correct / total_num
print('准确率为：', acc)

# 随机选一个batch（test_loader设置了shuffle为True），绘制其具体的测试结果图，看看机器队是否打败了人工队
x, y = next(iter(test_loader))
out = net(x.view(x.size(0), 28*28))
predict = out.argmax(dim=1)  # 返回最大值对应的标签
plot_image(x, predict, 'test_result')