import torch
from torch.utils.data import DataLoader

TRAIN_PATH = "data/processed/test.pt"
TEST_PATH = "data/processed/training.pt"

train_data = torch.load(TRAIN_PATH)
test_data = torch.load(TEST_PATH)

trainloader = DataLoader(train_data, batch_size=64, shuffle=True)
testloader = DataLoader(test_data, batch_size=64, shuffle=True)