from config import Config
from model import Roberta
import utils as utils
from train import Train
import warnings
import torch
import numpy as np
import random

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    config = Config()
    train = Train(config)
    model = Roberta(config).to(device=config.device)
    train_loader, test_loader = utils.load_data(config)
    model = train.train(model=model, train_loader=train_loader, test_loader=test_loader)


