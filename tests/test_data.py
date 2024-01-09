import torch
import os
import pytest

from tests import _PATH_DATA

TRAIN_PATH_DATA = _PATH_DATA + "processed/train_articles_tokens.pt"
N_train = 119924

@pytest.mark.skipif(not os.path.exists("processed"), reason="Data files not found")
def test_data():
    train_dataset = torch.load(TRAIN_PATH_DATA)
    assert len(train_dataset["attention_mask"]) == N_train, "Number of samples in train dataset is incorrect"
    assert train_dataset["attention_mask"][0].shape == torch.Size([1024]), "Attention mask shape is incorrect"

if __name__ == '__main__':
    test_data()