import torch
import os
import argparse


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

PATH_TO_DATASET = '../protoqa-data/data/dev/dev.crowdsourced.jsonl'
PATH_TO_MODELS = 'models'
PATH_TO_RESULTS = 'results'
TEST_SIZE = 0.2
LEARNING_RATE = 5e-5  # try and test
MOMENTUM = 0.0  # try and test
DEVICE = get_device()
NUM_EPOCHS = 100000
BATCH_SIZE = 2
NUM_CHOICES = 2
LOSS = 'CrossEntropy'
OPTIM = 'sgd'


def parse_args(arguments=None):
    parser = argparse.ArgumentParser(description="Generate cell segmentation dataset")
    parser.add_argument(
        "-ptd", "--path_to_dataset", default=PATH_TO_DATASET, help="path to the dataset"
    )

    parser.add_argument(
        "-ptr", "--path_to_results", default=PATH_TO_RESULTS, help="path to directory for saving the results"
    )

    parser.add_argument('-ts', '--test_size', default=TEST_SIZE, help="Size of the test set in the dataset")

    parser.add_argument('-nc', '--num_choices', default=NUM_CHOICES, help="Number of choices for answers")

    parser.add_argument('-lr', '--learning_rate', default=LEARNING_RATE, help="Learning rate for training the model")

    parser.add_argument('-m', '--momentum', default=MOMENTUM, help="Momentum value for the optimizer")

    parser.add_argument('-ne', '--num_epochs', default=NUM_EPOCHS, help="Number of epochs of training")

    parser.add_argument('-bs', '--batch_size', default=BATCH_SIZE, help="Batch size in train set")

    parser.add_argument('-pm', '--path_to_model', default=PATH_TO_MODELS, help="Path to the saved models")

    parser.add_argument('-ls', '--loss', default=LOSS, help="Loss for training model")

    parser.add_argument('-o', '--optimizer', default=OPTIM, help="Optimizer type")

    args = parser.parse_args(arguments)
    return args


class Config:
    def __init__(self):
        args = parse_args()
        self.path_to_dataset = args.path_to_dataset
        self.path_to_results = args.path_to_results
        self.device = DEVICE
        self.test_size = args.test_size
        self.lr = args.learning_rate
        self.momentum = args.momentum
        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size
        self.path_to_models = args.path_to_model
        self.num_choices = args.num_choices
        self.loss = args.loss
        self.optimizer = args.optimizer
        os.makedirs(self.path_to_models, exist_ok=True)
        os.makedirs(self.path_to_results, exist_ok=True)

    def set_number_of_epochs(self, value):
        self.num_epochs = value
