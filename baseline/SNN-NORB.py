import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import norse.torch as norse
import torchmetrics


class SNNetwork(torch.nn.Module):
    def __init__(self, input_features, num_classes):
        super(SNNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(input_features, 100)
        self.lif1 = norse.LIFCell()
        self.fc2 = torch.nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x, _ = self.lif1(x)
        x = self.fc2(x)
        return x

def read_norb_file(filepath, expected_dims):
    with open(filepath, 'rb') as file:
        file.read(4)  # skip magic number
        ndim = int.from_bytes(file.read(4), 'little')
        dims = [int.from_bytes(file.read(4), 'little') for _ in range(ndim)]

        assert dims == expected_dims, f"Dimensions mismatch, expected {expected_dims} but got {dims}"

        data = np.fromfile(file, dtype=np.uint8 if ndim == 4 else np.int32)
        if ndim == 4:
            data = data.reshape(dims)
        return data

import numpy as np

def read_norb_labels(filepath):
    with open(filepath, 'rb') as file:
        file.read(4)  # Skip magic number
        file.read(4)  # Skip ndim (dimension count)
        num_samples = int.from_bytes(file.read(4), 'little')  # Read number of samples

        # 跳过任何需要忽略的整数值
        file.read(4)  # Skip first redundant integer
        file.read(4)  # Skip second redundant integer

        # 读取实际的标签数据
        labels = np.fromfile(file, dtype=np.int32)
        return labels

def load_data(image_path, label_path):
    # Use previously defined functions
    images = read_norb_file(image_path, [24300, 2, 96, 96])
    labels = read_norb_labels(label_path)

    # Flatten images
    num_samples = images.shape[0]
    images_reshaped = images.reshape(num_samples, -1)

    # Normalize images
    images_normalized = images_reshaped.astype('float32') / 255.0

    # One-hot encode labels
    encoder = OneHotEncoder(sparse=False)
    labels_encoded = encoder.fit_transform(labels.reshape(-1, 1))

    # Convert to tensors
    features_tensor = torch.tensor(images_normalized, dtype=torch.float32)
    labels_tensor = torch.tensor(labels_encoded, dtype=torch.float32)

    return features_tensor, labels_tensor


def main():
    image_path = '/home/wyd/home/wyd/spikebls/smallNORB/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat'
    label_path = '//home/wyd/home/wyd/spikebls/smallNORB/smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat'
    features, labels = load_data(image_path, label_path)
    dataset = TensorDataset(features, torch.max(labels, 1)[1])
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = SNNetwork(96 * 96 * 2, 5)  # Assumes 5 classes
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    accuracy = torchmetrics.Accuracy(num_classes=5, average='macro', task='multiclass')

    model.train()
    for epoch in range(20):  # Train for 20 epochs
        for data, target in loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            accuracy(output, target)

        print(f'Epoch {epoch + 1}, Loss: {loss.item()}, Accuracy: {accuracy.compute()}')
        accuracy.reset()


if __name__ == '__main__':
    main()
