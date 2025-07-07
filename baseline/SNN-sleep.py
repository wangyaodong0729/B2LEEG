import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.io import loadmat
import norse.torch as norse
import torchmetrics

class SNNetwork(torch.nn.Module):
    def __init__(self, num_channels, num_classes):
        super(SNNetwork, self).__init__()
        self.conv1 = torch.nn.Conv1d(num_channels, 32, kernel_size=3, padding=1)
        self.lif1 = norse.LIFCell()
        self.pool1 = torch.nn.MaxPool1d(2)
        self.fc1 = torch.nn.Linear(1500 * 32, 100)
        self.fc2 = torch.nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x, _ = self.lif1(x)
        x = self.pool1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
def load_data(path):
    data = loadmat(path)
    features = torch.tensor(data['features'], dtype=torch.float32)  # Adjust for Conv1d (batch, channels, sequence_length)
    labels = torch.tensor(data['classes'].squeeze(), dtype=torch.long)
    # Remap label '5' to '4'
    labels[labels == 5] = 4
    return features, labels


    return features, labels

def main():
    features, labels = load_data('/home/wyd/spikebls/sleepS01.mat')
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = SNNetwork(6, 5)  # 6 channels, 5 classes
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
