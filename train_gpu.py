import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from model import *  # 导入网络模型
import time          # 记录训练用时

# 利用GPU训练，需要将 1.网络模型；2.数据（输入、标签）；3.损失函数 三者传入GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# OR device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# torchvision数据集加载完后的输出是范围在[0, 1]之间的PIL_Image，需要先将图片标准化为范围在[-1, 1]之间的张量
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
test_data = torchvision.datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 查看数据集大小
train_data_size = len(train_data)
test_data_size = len(test_data)

print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))


# 利用 DataLoader 来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True, num_workers=0)


# 实例化创建网络模型
cifar = Cifar10()

# 张量可以使用.to方法移动到任何设备device上
cifar = cifar.to(device)  # OR cifar.to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)  # OR loss_fn.to(device)

# 优化器
learning_rate = 1e-2   # 1e-2=1 x (10)^(-2) = 1 /100 = 0.01
optimizer = torch.optim.SGD(cifar.parameters(), lr=learning_rate) # cifar.parameters()获取模型所有参数


# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter("logs_cifar10_gpu")

start_time = time.time()
for i in range(epoch):
    print("-------第 {} 轮训练开始-------".format(i+1))

    # 训练步骤开始
    cifar.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs, targets = imgs.to(device), targets.to(device)
        outputs = cifar(imgs) # 前向传播
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad() # 将梯度缓存归零，防止累加
        loss.backward()       # 反向传播计算梯度
        optimizer.step()      # 利用梯度来更新模型参数

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print("训练用时：{}".format(end_time - start_time))
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 保存最后训练完的模型
    if i == 9:
        # 方式1：仅保留模型参数
        torch.save(cifar.state_dict(), "cifar.pth")

        # 方式2：同时保存模型结构和模型参数
        # torch.save(cifar, "cifar_2.pth")

        print("CIFAR10分类模型已保存")

    # 测试步骤开始
    cifar.eval()
    total_test_loss = 0
    total_accuracy = 0
    class_correct = list(0 for i in range(10))
    class_total = list(0 for i in range(10))
    # 测试时不再对模型梯度进行调优
    with torch.no_grad(): # torch.no_grad()将所有tensor的requires_grad都自动设置为False
        for data in test_dataloader:
            imgs, targets = data
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = cifar(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item() # 如果是仅包含一个元素的tensor，可以使用.item()来得到对应的数值
            accuracy = (outputs.argmax(1) == targets).sum() # 每一行最大值的索引与标签相等则预测正确
            total_accuracy += accuracy
            acc_tensor =  (outputs.argmax(1) == targets).squeeze()
            for i in range(16):
                taget= targets[i]
                class_correct[taget] += acc_tensor[i]
                class_total[taget] += 1


    print("整体测试集上的Loss: {}".format(total_test_loss))
    print("整体测试集上的正确率: {} %".format(100 * total_accuracy/test_data_size))
    for i in range(10):
        print('识别为{}的正确率为: {} %'.format(classes[i], 100 * class_correct[i] / class_total[i]))


    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step = total_test_step + 1


# 使用 TensorBoard 检查模型
writer.add_graph(cifar, imgs)
writer.close()
