from torch.utils.data import DataLoader
from datasets import load_from_disk
from os import path

# _DATA_PATH = 


class PaperDataLoader(DataLoader):
    def __init__(self, subset, batch_size = 64, shuffle=True, num_workers=0):
        """ Loads data from disk and creates a PyTorch DataLoader object.
        
            Args:
            subset: name of dataset subset (e.g. "train", "val", "test")
            batch_size: batch size
            shuffle: whether to shuffle the data
            num_workers: number of workers to use for loading data
        """
        self.folder = path.join(path.dirname(__file__), "..", "..", "data", "processed")
        
        # Load data from disk
        data = load_from_disk(path.join(self.folder, subset + "_data"))
        
        # Create DataLoader object
        super().__init__(data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


if __name__ == '__main__':
    DataLoader = PaperDataLoader("train", 2)
    for batch in DataLoader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        print(input_ids.shape)
        print(attention_mask.shape)
        print(labels.shape)
        break
        
