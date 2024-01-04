from datasets import load_dataset, load_dataset, get_dataset_split_names
from transformers import AutoTokenizer

TOKENIZER = AutoTokenizer.from_pretrained("bart-large-cnn")
DATA_NAME = "scientific_papers"
SUBSET = "pubmed"
MAX_LENGTH = 512

def tokenization(data: dict, which: str ="article") -> dict:
    ''' Tokenizes the data and adds special tokens for BERT to work properly. '''
    return TOKENIZER(data[which], truncation=True, padding="max_length", max_length=MAX_LENGTH)

def fetch_data() -> dict:
    ''' Fetches the dataset from the HuggingFace library. '''
    return load_dataset(DATA_NAME, SUBSET)

def preprocess(dataset: dict) -> tuple(dict, dict):
    ''' Preprocesses the dataset by tokenizing the articles and abstracts. '''
    articles = dataset.map(tokenization, batched=True, fn_kwargs={"which": "article"})
    abstracts = dataset.map(tokenization, batched=True, fn_kwargs={"which": "abstract"})

    articles.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask"])
    abstracts.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask"])

    return articles, abstracts

def split_data(dataset):
    train, val, test = get_dataset_split_names(DATA_NAME, SUBSET)
    train_data = dataset[train]
    val_data = dataset[val]
    test_data = dataset[test]
    return train_data, val_data, test_data

if __name__ == '__main__':
    print("Fetching data...")
    dataset = fetch_data()

    print("Splitting data...")
    train_data, val_data, test_data = split_data(dataset)

    print("Preprocessing data...")
    # train_articles, train_abstracts = preprocess(train_data)
    val_articles, val_abstracts = preprocess(val_data)
    test_articles, test_abstracts = preprocess(test_data)

    print("saving data...")
    # train_articles.save_to_disk("data/processed/train_articles")
    # train_abstracts.save_to_disk("data/processed/train_abstracts")
    val_articles.save_to_disk("data/processed/val_articles")
    val_abstracts.save_to_disk("data/processed/val_abstracts")
    test_articles.save_to_disk("data/processed/test_articles")
    test_abstracts.save_to_disk("data/processed/test_abstracts")

    
