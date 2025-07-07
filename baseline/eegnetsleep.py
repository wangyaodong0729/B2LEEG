import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.io import loadmat
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

class EEGNet(nn.Module):
    def __init__(self, num_channels, num_samples, num_classes):
        super(EEGNet, self).__init__()
        # First Block
        self.firstConv = nn.Conv2d(1, 16, (1, 51), padding=(0, 25), bias=False)
        self.batchNorm1 = nn.BatchNorm2d(16)
        self.depthwiseConv = nn.Conv2d(16, 16 * 2, (num_channels, 1), groups=16, bias=False)
        self.batchNorm2 = nn.BatchNorm2d(16 * 2)
        self.activation = nn.ELU()
        self.pooling1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(0.5)
        # Second Block
        self.separableConv = nn.Conv2d(16 * 2, 16 * 4, (1, 15), padding=(0, 7), bias=False)
        self.batchNorm3 = nn.BatchNorm2d(16 * 4)
        self.pooling2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(0.5)
        # Classification layer
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(16 * 4 * (num_samples // 32), num_classes)

    def forward(self, x):
        x = self.firstConv(x)
        x = self.batchNorm1(x)
        x = self.depthwiseConv(x)
        x = self.batchNorm2(x)
        x = self.activation(x)
        x = self.pooling1(x)
        x = self.dropout1(x)
        x = self.separableConv(x)
        x = self.batchNorm3(x)
        x = self.activation(x)
        x = self.pooling2(x)
        x = self.dropout2(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

def load_data(path):
    data = loadmat(path)
    features = torch.tensor(data['features'], dtype=torch.float32).unsqueeze(1)  # [batch, 1, channels, samples]
    labels = torch.tensor(data['classes'].squeeze(), dtype=torch.long)
    labels[labels == 5] = 4
    return features, labels

def main():
    features, labels = load_data('/home/wyd/spikebls/sleepS01.mat')
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = EEGNet(6, 3000, 5)  # 6 channels, 3000 samples, 5 classes
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    accuracy = torchmetrics.Accuracy(num_classes=5, average='macro', task='multiclass')
    recall = torchmetrics.Recall(average='macro', num_classes=5, task='multiclass')
    f1_score = torchmetrics.F1Score(average='macro', num_classes=5, task='multiclass')

    model.train()
    for epoch in range(50):
        for data, target in loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(output, dim=1)
            accuracy.update(preds, target)
            recall.update(preds, target)
            f1_score.update(preds, target)

        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
        print(f'Accuracy: {accuracy.compute()}')
        print(f'Recall: {recall.compute()}')
        print(f'F1 Score: {f1_score.compute()}')

        accuracy.reset()
        recall.reset()
        f1_score.reset()

if __name__ == '__main__':
    main()

