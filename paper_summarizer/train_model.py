import torch
from torch.utils.data import DataLoader
import hydra
import wandb
import omegaconf
import evaluate
import numpy as np
from model import ScientificPaperSummarizer
from transformers import get_scheduler
from tqdm.auto import tqdm
from torch.optim import AdamW
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq

## using https://huggingface.co/docs/transformers/tasks/summarization#train

def compute_metrics(eval_pred):
    #evaluation metric
    rouge = evaluate.load("rouge")
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

def train_model(train_data_pt, eval_data_pt, model, num_epochs=4):
    
    # Create a DataCollatorForSeq2Seq instance
    data_collator = DataCollatorForSeq2Seq(model.tokenizer, model=model)

    # Create a DataLoader
    train_dataloader = DataLoader(train_data_pt, batch_size=64, collate_fn=data_collator)
    eval_dataloader = DataLoader(eval_data_pt, batch_size=64, collate_fn=data_collator)
    
    # Training argument to contain hyperparameters
    # default optimizer is Adam optimizer
    training_args = Seq2SeqTrainingArguments(
        output_dir="fine_tuned_summarizer",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=num_epochs,
        predict_with_generate=True,
        fp16=True,
        push_to_hub=True,
        )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataloader,
        eval_dataset=eval_dataloader,
        tokenizer=model.tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics, #rouge evaluation metric
        )

    trainer.train()
        


@hydra.main(config_name="config.yaml", config_path="../configs/")
def main(cfg):
    cfg = cfg.experiments
    
    # setup wandb
    wandb_config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    wandb_run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project, config=wandb_config)
    
    
    TRAIN_PATH = "data/processed/train_articles_tokens.pt"
    TEST_PATH = "data/processed/val_articles_tokens.ptt"

    train_data = torch.load(TRAIN_PATH)
    eval_data = torch.load(TEST_PATH)
    
    model = ScientificPaperSummarizer()
    train_model(train_data, eval_data, model)
    

#trainloader = DataLoader(train_data, batch_size=64, shuffle=True)
#testloader = DataLoader(test_data, batch_size=64, shuffle=True)

if __name__ == "__main__":
    main()
