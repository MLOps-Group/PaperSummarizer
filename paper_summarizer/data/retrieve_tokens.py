
from datasets import load_from_disk
from tqdm import tqdm
import torch
def retrieve_tokens(row):
        ''' Retrieves the tokens from the dataset. '''
        return row['input_ids'], row['attention_mask']

if __name__ == '__main__':
    train_articles = load_from_disk("data/processed/train_articles")
    train_abstracts = load_from_disk("data/processed/train_abstracts")

    val_articles = load_from_disk("data/processed/val_articles")
    val_abstracts = load_from_disk("data/processed/val_abstracts")

    test_articles = load_from_disk("data/processed/test_articles")
    test_abstracts = load_from_disk("data/processed/test_abstracts")

    print("Retrieving tokens...")

    paths = ["data/processed/train_articles_tokens.pt",
             "data/processed/train_abstracts_tokens.pt",
             "data/processed/val_articles_tokens.pt",
             "data/processed/val_abstracts_tokens.pt",
             "data/processed/test_articles_tokens.pt",
             "data/processed/test_abstracts_tokens.pt"]
    
    datasets = [train_articles, train_abstracts, val_articles, val_abstracts, test_articles, test_abstracts]

    for i in range(len(paths)):
        tokens = []
        attentions = []
        for row in tqdm(datasets[i]):
            tokens.append(torch.tensor(row['input_ids']))
            attentions.append(torch.tensor(row['attention_mask']))
        torch.save({"input_ids": tokens, "attention_mask": attentions}, paths[i])




