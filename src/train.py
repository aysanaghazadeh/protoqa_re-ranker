from torch.optim import SGD, Adam
from torch.nn import CrossEntropyLoss, HingeEmbeddingLoss
from torch import nn
from tqdm import tqdm
from transformers import RobertaTokenizer
import torch
import numpy as np


class Train(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loss = CrossEntropyLoss() if self.config == 'CrossEntropy' else HingeEmbeddingLoss()
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.results = {"train_loss": [], "test_loss": []}
        self.epoch = self.config.num_epochs

    def train(self, model, train_loader, test_loader):
        optimizer = \
            SGD(model.parameters(), lr=self.config.lr, momentum=self.config.momentum) \
                if self.config.optimizer == 'sgd' \
                else Adam(model.parameters(), lr=self.config.lr, momentum=self.config.momentum)

        for epoch in tqdm(range(self.epoch)):
            model.train()
            total_loss = 0
            preds = []
            labels = []
            for (i, (question, choices, label)) in enumerate(train_loader):
                encoding = self.tokenizer([question, question], choices, return_tensors="pt", padding=True)
                encoding, label = encoding.to(device=self.config.device), label.to(device=self.config.device)
                prediction = model(encoding)
                preds.append(torch.argmax(prediction.cpu()))
                labels.append(label)
                loss = self.loss(prediction, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss
            print("[INFO] EPOCH: {}/{}".format(epoch + 1, self.config.num_epochs))
            print("Train loss: {:.6f}".format(total_loss/i))
            print("Train Accuracy: {:.6f}".format(np.sum(np.array(labels) == np.array(preds))/len(preds)))
            labels = []
            preds = []
            for (i, (question, choices, label)) in enumerate(test_loader):
                encoding = self.tokenizer([question, question], choices, return_tensors="pt", padding=True)
                encoding, label = encoding.to(device=self.config.device), label.to(device=self.config.device)
                prediction = model(encoding)
                preds.append(torch.argmax(prediction.cpu()))
                labels.append(label)
            print("Test Accuracy: {:.6f}".format(np.sum(np.array(labels) == np.array(preds)) / len(preds)))
