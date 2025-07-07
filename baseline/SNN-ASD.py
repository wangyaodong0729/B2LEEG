import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.io import loadmat
import norse.torch as norse
import torchmetrics
class SNNetwork(torch.nn.Module):
    def __init__(self, input_features, num_classes):
        super(SNNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(input_features, 100)  # 100可以根据需要调整
        self.lif1 = norse.LIFCell()  # 利用Norse库中的LIF模型
        self.fc2 = torch.nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x, _ = self.lif1(x)
        x = self.fc2(x)
        return x

def load_data(path):
    data = loadmat(path)
    features = torch.tensor(data['features'], dtype=torch.float32)
    labels = torch.tensor(data['classes'].squeeze(), dtype=torch.long)
    return features, labels

def main():
    features, labels = load_data('/home/wyd/spikebls/PSDfeatures.mat')
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = SNNetwork(40, 2)  # 假设有10个类
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    accuracy = torchmetrics.Accuracy(num_classes=2, average='macro', task='binary')
    recall = torchmetrics.Recall(average='macro', num_classes=2, task='binary')
    f1_score = torchmetrics.F1Score(average='macro', num_classes=2, task='binary')
    model.train()
    for epoch in range(20):  # 训练10个周期
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
