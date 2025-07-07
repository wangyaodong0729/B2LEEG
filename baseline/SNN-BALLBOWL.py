import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.io import loadmat
import norse.torch as norse
import torchmetrics

class SNNetwork(torch.nn.Module):
    def __init__(self, input_features, num_classes):
        super(SNNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(input_features, 50)  # 从100调整为50，更适合更小的输入尺寸
        self.lif1 = norse.LIFCell()
        self.fc2 = torch.nn.Linear(50, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x, _ = self.lif1(x)
        x = self.fc2(x)
        return x

def load_data(path):
    data = loadmat(path)
    features = torch.tensor(data['features'], dtype=torch.float32)
    # 确保标签是长整型
    labels = torch.tensor(data['classes'].squeeze(), dtype=torch.long)
    return features, labels

def main():
    features, labels = load_data('/home/wyd/spikebls/ball-bowl.mat')
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = SNNetwork(14, 2)  # 输入特征数为14，假设有10个类
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Metrics
    accuracy = torchmetrics.Accuracy(num_classes=2, average='macro', task='binary')
    recall = torchmetrics.Recall(average='macro', num_classes=2, task='binary')
    f1_score = torchmetrics.F1Score(average='macro', num_classes=2, task='binary')

    model.train()
    for epoch in range(10):  # 训练10个周期
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
