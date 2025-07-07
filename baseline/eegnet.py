import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.io import loadmat
import torchmetrics
import os
from PIL import Image
import numpy as np
class EEGNet1D(nn.Module):
    def __init__(self, num_classes=10):
        super(EEGNet1D, self).__init__()
        self.firstconv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ELU(),
            nn.Dropout(0.25)
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, groups=16, padding=1),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.AvgPool1d(kernel_size=4),
            nn.Dropout(0.25)
        )
        self.separableConv = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.AvgPool1d(kernel_size=4),
            nn.Dropout(0.25)
        )
        self.classify = nn.Sequential(
            nn.Linear(32 * 2, num_classes)
        )

    def forward(self, x):
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.classify(x)
        return x

def load_data(path):
    data = loadmat(path)
    features = torch.tensor(data['features'], dtype=torch.float32)
    labels = torch.tensor(data['classes'].squeeze(), dtype=torch.int64)
    # Reshape features for Conv1D input
    features = features.unsqueeze(1)  # Shape: (batch_size, 1, num_features)
    return features, labels


def main():
    # 设置数据目录


    features, labels = load_data('/home/wyd/spikebls/combined_dataset.mat')

    # Create data loader
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    net = EEGNet1D(num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)


    # Metrics
    accuracy = torchmetrics.Accuracy(num_classes=2, average='macro', task='binary')
    recall = torchmetrics.Recall(average='macro', num_classes=2, task='binary')
    f1_score = torchmetrics.F1Score(average='macro', num_classes=2, task='binary')


    # Train the network
    net.train()
    for epoch in range(10):
        for data, target in loader:
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Update metrics
            preds = torch.argmax(output, dim=1)
            accuracy.update(preds, target)
            recall.update(preds, target)
            f1_score.update(preds, target)

        # Print metrics
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
        print(f'Accuracy: {accuracy.compute()}')
        print(f'Recall: {recall.compute()}')
        print(f'F1 Score: {f1_score.compute()}')

        # Reset metrics for next epoch
        accuracy.reset()
        recall.reset()
        f1_score.reset()


if __name__ == '__main__':
    main()