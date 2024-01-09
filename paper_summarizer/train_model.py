import torch
from torch.utils.data import DataLoader
import hydra
import wandb
import omegaconf
import os
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer


@hydra.main(config_name="config.yaml", config_path="../configs/")
def main(cfg):
    cfg = cfg.experiments
    
    # setup wandb
    wandb_config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    wandb_run = wandb.init(**cfg.wandb, config=wandb_config)
    print(wandb_run.config)


    # setup model
    model = AutoModelForSeq2SeqLM.from_pretrained(wandb_run.config.model_name) # download and load the pretrained model
    tokenizer = AutoTokenizer.from_pretrained(wandb_run.config.model_name) #load the tokenizer related to the model
    
    # setup dataset
    training_data_tokenized = None
    validation_data_tokenized = None
    
    
    # setup training args
    os.makedirs(wandb_run.config.train_args.output_dir, exist_ok=True)
    training_args = Seq2SeqTrainingArguments(**wandb_run.config.train_args)

    # setup trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=training_data_tokenized,
        eval_dataset=validation_data_tokenized,
        tokenizer=tokenizer,
        callbacks=[wandb.run.summary],
        # data_collator=data_collator,
    )

    trainer.train()

#TRAIN_PATH = "data/processed/test.pt"
#TEST_PATH = "data/processed/training.pt"

#train_data = torch.load(TRAIN_PATH)
#test_data = torch.load(TEST_PATH)

#trainloader = DataLoader(train_data, batch_size=64, shuffle=True)
#testloader = DataLoader(test_data, batch_size=64, shuffle=True)

if __name__ == "__main__":
    main()
