from datasets import load_dataset
from transformers import BartTokenizer, logging
from typing import List, Tuple

logging.set_verbosity_error()

TOKENIZER = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

DATA_NAME = "scientific_papers"
SUBSET = "pubmed"
MAX_LENGTH = 1024


def tokenizer(examples):
    ''' Tokenizes the articles and abstracts. '''
    return TOKENIZER(text = examples["article"], text_target = examples["abstract"], truncation=True, padding="max_length", max_length=MAX_LENGTH, return_tensors="pt")

def preprocess(dataset: dict) -> Tuple[List[dict], List[dict]]:
    ''' Preprocesses the dataset by tokenizing the articles and abstracts. '''
    tokenized_dataset = dataset.map(tokenizer, batched=True).remove_columns(["article", "abstract","section_names"])
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized_dataset

if __name__ == '__main__':
    print("Fetching data...")
    train_data = load_dataset(DATA_NAME, SUBSET, split="train")
    val_data = load_dataset(DATA_NAME, SUBSET, split="validation")
    test_data = load_dataset(DATA_NAME, SUBSET, split="test")

    print("Preprocessing validation data...")
    val_data = preprocess(val_data)
    print("Saving data...")
    val_data.save_to_disk("data/processed/val_data")
    del val_data

    print("Preprocessing test data...")
    test_data = preprocess(test_data)
    print("Saving data...")
    test_data.save_to_disk("data/processed/test_data")
    del test_data

    print("Preprocessing training data...")
    train_data = preprocess(train_data)
    print("Saving data...")
    train_data.save_to_disk("data/processed/train_data")
    del train_data

    
