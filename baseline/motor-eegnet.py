import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.io import loadmat
import torchmetrics

class EEGNet1D(nn.Module):
    def __init__(self):
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
            nn.AvgPool1d(kernel_size=2),
            nn.Dropout(0.25)
        )
        self.separableConv = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.AvgPool1d(kernel_size=2),
            nn.Dropout(0.25)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 3, 2)  # Adjusted for the expected output size
        )

    def forward(self, x):
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def load_data(path):
    data = loadmat(path)
    features = torch.tensor(data['features'], dtype=torch.float32)
    labels = torch.tensor(data['classes'], dtype=torch.float32)  # Assuming one-hot encoded labels
    features = features.unsqueeze(1)  # Adding channel dimension
    return features, labels

def main():
    features, labels = load_data('/home/wyd/spikebls/ball-bowl.mat')
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    net = EEGNet1D()
    criterion = nn.BCEWithLogitsLoss()  # Suitable for binary classification with one-hot labels
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # Metrics
    accuracy = torchmetrics.Accuracy(threshold=0.0,task='binary')
    recall = torchmetrics.Recall(num_classes=2, average='macro', task='binary',)
    f1 = torchmetrics.F1Score(num_classes=2, average='macro', task='binary',)

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
            preds = torch.sigmoid(output)  # Apply sigmoid to get probabilities
            preds = preds > 0.5  # Convert probabilities to binary output
            accuracy.update(preds, target.int())
            recall.update(preds, target.int())
            f1.update(preds, target.int())

        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
        print(f'Accuracy: {accuracy.compute()}')
        print(f'Recall: {recall.compute()}')
        print(f'F1 Score: {f1.compute()}')

        # Reset metrics for the next epoch
        accuracy.reset()
        recall.reset()
        f1.reset()

if __name__ == '__main__':
    main()
