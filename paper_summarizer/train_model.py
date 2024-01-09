import torch
from torch.utils.data import DataLoader
import hydra
import wandb
import omegaconf
# import evaluate
import numpy as np

from transformers import get_scheduler
from tqdm.auto import tqdm
from torch.optim import AdamW
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import os
from paper_summarizer.data.dataloader import PaperDataLoader
from paper_summarizer.models.model import ScientificPaperSummarizer

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

def train_model(model, training_args):
    
    # Create a DataCollatorForSeq2Seq instance
    data_collator = DataCollatorForSeq2Seq(model.tokenizer, model=model)

    # Create a DataLoader
    train_dataloader = PaperDataLoader("train", batch_size=64, shuffle=True, data_path="../../data/processed/")
    val_dataloader = PaperDataLoader("val", batch_size=64, shuffle=False, data_path="../../data/processed/")
    
    
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataloader,
        eval_dataset=val_dataloader,
        tokenizer=model.tokenizer,
        data_collator=data_collator,
        # compute_metrics=compute_metrics, #rouge evaluation metric
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

    # setup model    
    model = ScientificPaperSummarizer()
    
    # Training argument to contain hyperparameters
    # default optimizer is Adam optimizer
    # setup training args
    os.makedirs(wandb_run.config.train_args["output_dir"], exist_ok=True)
    training_args = Seq2SeqTrainingArguments(**wandb_run.config.train_args)
    
    # train model
    train_model(model, training_args)
    

#trainloader = DataLoader(train_data, batch_size=64, shuffle=True)
#testloader = DataLoader(test_data, batch_size=64, shuffle=True)

if __name__ == "__main__":
    main()
