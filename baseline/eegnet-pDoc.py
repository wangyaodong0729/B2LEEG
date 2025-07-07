import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchmetrics
import pandas as pd
import numpy as np


class EEGNet(nn.Module):
    def __init__(self, num_classes, channels=1, samples=1170):
        super(EEGNet, self).__init__()

        # First convolutional block
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(16)
        )

        # Depthwise convolutional block
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, (channels, 1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.25)
        )

        # Separable convolutional block
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, (1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.25)
        )

        # Placeholder for the final linear layer, actual size will be set dynamically
        self.classify = nn.Linear(32 * 18, num_classes)



    def forward(self, x):
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = torch.flatten(x, 1)
        return self.classify(x)


def main():
    # Create data loader
    df = pd.read_csv('/home/wyd/spikebls/pdoc.csv')
    features = df.iloc[1:, :-10].to_numpy()
    labels = df.iloc[1:, -1].to_numpy()

    # Convert to numeric labels
    labels = pd.get_dummies(labels).to_numpy()
    labels = np.array(labels, dtype=np.float32)
    labels = np.argmax(labels, axis=1)

    # Convert to tensors
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.int64)

    # Reshape features for EEGNet model: (N, 1, 1, samples)
    features = features.view(-1, 1, 1, features.shape[1])

    # Create dataset and loader
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Initialize EEGNet model
    model = EEGNet(num_classes=3, channels=1, samples=features.shape[3])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    accuracy = torchmetrics.Accuracy(num_classes=3, average='macro', task='multiclass')
    recall = torchmetrics.Recall(average='macro', num_classes=3, task='multiclass')
    f1_score = torchmetrics.F1Score(average='macro', num_classes=3, task='multiclass')

    # Training loop
    model.train()
    for epoch in range(50):
        for data, target in loader:
            optimizer.zero_grad()
            output = model(data)
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
