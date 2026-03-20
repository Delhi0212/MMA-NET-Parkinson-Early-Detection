import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ==========================================
# DATASET CLASS
# ==========================================

class ParkinsonDataset(Dataset):

    def __init__(self, size=500):

        self.size = size

        # Voice features (22)
        self.voice = np.random.rand(size, 22)

        # Gait sequence (300 time steps, 16 features)
        self.gait = np.random.rand(size, 300, 16)

        # Labels (0 = Healthy, 1 = PD)
        self.labels = np.random.randint(0, 2, size)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):

        sample = {
            "voice": torch.tensor(self.voice[idx], dtype=torch.float32),
            "gait": torch.tensor(self.gait[idx], dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

        return sample


# ==========================================
# DATALOADER FUNCTION
# ==========================================

def get_dataloaders():

    train_dataset = ParkinsonDataset(size=400)
    test_dataset = ParkinsonDataset(size=100)

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False
    )

    return train_loader, test_loader
