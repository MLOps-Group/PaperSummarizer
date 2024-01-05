import torch
from torch.utils.data import DataLoader
import hydra
import wandb
import omegaconf

# import logging
# log = logging.getLogger(__name__)


@hydra.main(config_name="config.yaml", config_path="../configs/")
def main(cfg):
    cfg = cfg.experiments
    
    # setup wandb
    wandb_config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    wandb_run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project, config=wandb_config)




#TRAIN_PATH = "data/processed/test.pt"
#TEST_PATH = "data/processed/training.pt"

#train_data = torch.load(TRAIN_PATH)
#test_data = torch.load(TEST_PATH)

#trainloader = DataLoader(train_data, batch_size=64, shuffle=True)
#testloader = DataLoader(test_data, batch_size=64, shuffle=True)

if __name__ == "__main__":
    main()
