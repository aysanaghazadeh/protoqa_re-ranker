from config import Config
from model import Roberta
import utils as utils
from train import Train
import warnings

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = Config()
    train = Train(config)
    model = Roberta(config).to(device=config.device)
    train_loader, test_loader = utils.load_data(config)
    model = train.train(model=model, train_loader=train_loader, test_loader=test_loader)


