import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.io import loadmat
import pandas as pd
import scipy.io as scio

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 120, kernel_size=5),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 3)
        )

    def forward(self, x):
        x = self.convnet(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def load_data(path):
    data = loadmat(path)
    features = torch.Tensor(data['features'])
    labels = torch.Tensor(data['labels']).long()
    return features, labels


def main():
    # 加载数据
    df = pd.read_csv('/home/wyd/spikebls/pdoc.csv')
    features = df.iloc[1:, :-10].to_numpy()
    trainlabel1_series = df.iloc[1:, -1]
    labels = pd.get_dummies(trainlabel1_series).to_numpy()
    # features, labels = load_data('path_to_your_mat_file.mat')

    # 调整数据维度以匹配网络（假设是灰度图像）
    # features = features.unsqueeze(1)  # 添加一个通道维度

    # 创建数据加载器
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 初始化网络和优化器
    net = LeNet5()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    # 训练网络
    for epoch in range(10):  # 训练10个周期
        for data, target in loader:
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')


if __name__ == '__main__':
    main()
