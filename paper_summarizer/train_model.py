import torch
from torch.utils.data import DataLoader
import hydra

import logging
log = logging.getLogger(__name__)


@hydra.main(config_name="config.yaml", config_path="../conf/")
def main(cfg):
    log.info("Batch size: " + str(cfg.hyperparameters.batch_size))
    log.info("Learning Rate: " + str(cfg.hyperparameters.learning_rate))


#TRAIN_PATH = "data/processed/test.pt"
#TEST_PATH = "data/processed/training.pt"

#train_data = torch.load(TRAIN_PATH)
#test_data = torch.load(TEST_PATH)

#trainloader = DataLoader(train_data, batch_size=64, shuffle=True)
#testloader = DataLoader(test_data, batch_size=64, shuffle=True)

if __name__ == "__main__":
    main()
