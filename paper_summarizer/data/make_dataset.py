from datasets import load_dataset
from transformers import BartTokenizer
from tqdm import tqdm
from typing import List, Tuple
TOKENIZER = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

DATA_NAME = "scientific_papers"
SUBSET = "pubmed"
MAX_LENGTH = 1024

def preprocess(dataset: dict) -> Tuple[List[dict], List[dict]]:
    ''' Preprocesses the dataset by tokenizing the articles and abstracts. '''
    articles = dataset.map(lambda x: TOKENIZER(x["article"],
                                               return_tensors="pt",     # Save as PyTorch tensors
                                               truncation=True,         # Truncate to max_length
                                               padding="max_length",    # Pad to max_length
                                               max_length=MAX_LENGTH    # Max length of 1024
                                               ), batched=True) 
    
    abstracts = dataset.map(lambda x: TOKENIZER(x["abstract"],
                                                return_tensors="pt",
                                               truncation=True,
                                               padding="max_length",
                                               max_length=MAX_LENGTH
                                               ), batched=True)
    return articles, abstracts


if __name__ == '__main__':
    print("Fetching data...")
    train_data = load_dataset(DATA_NAME, SUBSET, split="train")
    val_data = load_dataset(DATA_NAME, SUBSET, split="validation")
    test_data = load_dataset(DATA_NAME, SUBSET, split="test")

    print("Preprocessing training data...")
    train_articles, train_abstracts = preprocess(train_data)

    print("Preprocessing validation data...")
    val_articles, val_abstracts = preprocess(val_data)

    print("Preprocessing test data...")
    test_articles, test_abstracts = preprocess(test_data)

    print("saving data...")
    train_articles.save_to_disk("data/raw/train_articles")
    train_abstracts.save_to_disk("data/raw/train_abstracts")
    val_articles.save_to_disk("data/raw/val_articles")
    val_abstracts.save_to_disk("data/raw/val_abstracts")
    test_articles.save_to_disk("data/raw/test_articles")
    test_abstracts.save_to_disk("data/raw/test_abstracts")

    
