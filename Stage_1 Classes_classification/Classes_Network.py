import torch.nn as nn
import torch.nn.functional as F


# 用于“纲”分类的网络。
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 进行2维的卷积(输入3个通道、输出3个通道、卷积核大小是3*3)
        self.conv1 = nn.Conv2d(3, 3, 3)
        # 进行max-pooling 窗口大小是2
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        # 整流,小于0的设为0
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(3, 6, 3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.relu2 = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(6 * 123 * 123, 150)
        self.relu3 = nn.ReLU(inplace=True)
        # 随机将整个通道归零（通道是2D特征图，
        self.drop = nn.Dropout2d()

        self.fc2 = nn.Linear(150, 2)
        self.softmax1 = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.relu2(x)

        # print(x.shape)
        x = x.view(-1, 6 * 123 * 123)
        x = self.fc1(x)
        x = self.relu3(x)

        x = F.dropout(x, training=self.training)

        x_classes = self.fc2(x)
        x_classes = self.softmax1(x_classes)

        return x_classes