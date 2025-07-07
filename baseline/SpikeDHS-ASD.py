import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.io import loadmat
import torch.nn as nn
import torch.optim as optim
import norse.torch as norse
import torchmetrics
import torch.nn.functional as F
class PlasticSynapse(nn.Module):
    def __init__(self, input_features, output_features):
        super(PlasticSynapse, self).__init__()
        self.weight = nn.Parameter(torch.randn(output_features, input_features) * 0.01)
        self.alpha = nn.Parameter(torch.randn(output_features, input_features) * 0.01)  # 可塑性系数

    def forward(self, x, pre_synaptic_activity):
        x_expanded = x.unsqueeze(2)  # [batch_size, features, 1]
        pre_act_expanded = pre_synaptic_activity.unsqueeze(1)  # [batch_size, 1, features]
        hebbian_update = torch.bmm(pre_act_expanded, x_expanded)  # [batch_size, 1, 1]
        self.weight.data += self.alpha * hebbian_update.mean(0)
        return F.linear(x, self.weight)

class PlasticSNN(nn.Module):
    def __init__(self, input_features, num_classes):
        super(PlasticSNN, self).__init__()
        self.fc1 = PlasticSynapse(input_features, 100)
        self.lif1 = norse.LIFCell()
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.fc1(x, x)
        x, _ = self.lif1(x)
        x = self.fc2(x)
        return x

def load_data(path):
    data = loadmat(path)
    features = torch.tensor(data['features'], dtype=torch.float32)
    labels = torch.tensor(data['classes'].squeeze(), dtype=torch.long)
    return features, labels


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features, labels = load_data('/home/wyd/spikebls/PSDfeatures.mat')
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = PlasticSNN(40, 2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Initialize metrics and move them to the appropriate device
    accuracy_metric = torchmetrics.Accuracy(num_classes=2, average='macro', task='binary').to(device)
    recall_metric = torchmetrics.Recall(num_classes=2, average='macro', task='binary').to(device)
    f1_score_metric = torchmetrics.F1Score(num_classes=2, average='macro', task='binary').to(device)

    for epoch in range(20):
        model.train()
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Update metrics with predictions and targets
            preds = torch.argmax(output, dim=1)
            accuracy_metric.update(preds, target)
            recall_metric.update(preds, target)
            f1_score_metric.update(preds, target)

        # Compute and print metrics at the end of each epoch
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
        print(f'Accuracy: {accuracy_metric.compute()}')
        print(f'Recall: {recall_metric.compute()}')
        print(f'F1 Score: {f1_score_metric.compute()}')

        # Reset metrics for next epoch
        accuracy_metric.reset()
        recall_metric.reset()
        f1_score_metric.reset()


if __name__ == '__main__':
    main()


