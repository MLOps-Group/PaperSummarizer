from torch.utils.data import DataLoader, Dataset
import torch
from os import path

_DATA_PATH = "data/processed/"


class PaperDataset(Dataset):
    
X = torch.load(path.join(_DATA_PATH, "val_articles_tokens.pt"))

print(X)