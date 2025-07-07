import os
from custom import CustomDataset
from torcheeg.model_selection import train_test_split, train_test_split_per_subject_cross_trial
from torcheeg import transforms
from torch.utils.data import DataLoader
import scipy.io as sio

if __name__ == "__main__":
    dataset = CustomDataset(
        io_path="./data_io",
        root_path="./ASD-gao",
        num_channel=8,
        offline_transform=transforms.BandDifferentialEntropy(
            band_dict={
                "delta": [1, 4],
                "theta": [4, 8],
                "alpha": [8, 14],
                "beta": [14, 31],
                "gamma": [31, 51],
            }
        ),
        online_transform=transforms.ToTensor(),
        num_worker=16,
    )

    train_dataset, val_dataset = train_test_split(
        dataset, shuffle=True, split_path="./spiltASD"
    )
    # train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)
    # train_dataset, val_dataset = train_test_split_per_subject_cross_trial(dataset=dataset, split_path='/home/wyd/spikebls/ASDspilt')

