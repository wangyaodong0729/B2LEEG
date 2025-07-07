import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.io import loadmat
import norse.torch as norse
import torchmetrics
import pandas as pd
import numpy as np
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
    labels = torch.tensor(data['classes'], dtype=torch.float32)
    return features, labels

def main():
    df = pd.read_csv('/home/wyd/spikebls/pdoc.csv')
    features = df.iloc[1:, :-10].to_numpy()
    labels = df.iloc[1:, -1].to_numpy()
    # Assuming labels is an `np.ndarray` that may contain non-numeric values


    labels = pd.get_dummies(labels).to_numpy()
    labels = np.array(labels, dtype=np.float32)
    labels = np.argmax(labels, axis=1)
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.int64)

    # features, labels = load_data('/home/wyd/spikebls/bci2b-f.mat')
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = SNNetwork(1170, 3)  # 假设有10个类
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    accuracy = torchmetrics.Accuracy(num_classes=3, average='macro', task='multiclass')
    recall = torchmetrics.Recall(average='macro', num_classes=3, task='multiclass')
    f1_score = torchmetrics.F1Score(average='macro', num_classes=3, task='multiclass')

    model.train()
    for epoch in range(50):  # 训练10个周期
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
