from torch.optim import SGD, Adam
from transformers import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.nn import CrossEntropyLoss, HingeEmbeddingLoss
from torch import nn
from tqdm import tqdm
import torch
import numpy as np


class Train(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loss = CrossEntropyLoss() if self.config.loss == 'CrossEntropy' else HingeEmbeddingLoss()
        self.results = {"train_loss": [], "test_loss": []}
        self.epoch = self.config.num_epochs
        self.softmax = nn.Softmax()

    def train(self, model, train_loader, test_loader):
        for param in model.parameters():
            param.requires_grad = True
        optimizer = \
            SGD(model.parameters(), lr=self.config.lr, momentum=self.config.momentum) \
                if self.config.optimizer == 'sgd' \
                else AdamW(model.parameters(), lr=self.config.lr)
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        for epoch in tqdm(range(self.epoch)):
            model.train()
            total_loss = 0
            preds = []
            labels = []
            for (i, (encoding, label)) in enumerate(train_loader):
                encoding = {k: v.to(device=self.config.device) for k, v in encoding.items()}
                label = label.to(self.config.device)
                prediction, loss = model(encoding, label)
                optimizer.zero_grad()
                if self.config.loss == 'CrossEntropy':
                    predicted_label = self.softmax(prediction)
                else:
                    predicted_label = prediction
                predicted_label = torch.argmax(predicted_label, dim=1)
                preds += predicted_label.cpu()
                labels += label.cpu()
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss
            print("[INFO] EPOCH: {}/{}".format(epoch + 1, self.config.num_epochs))
            print("Train loss: {}".format(total_loss/(len(preds)/self.config.batch_size)))
            print("Train Accuracy: {}".format(np.sum(np.array(labels) == np.array(preds))/len(preds)))
            # labels = []
            # preds = []
            # for (i, (encoding, label)) in enumerate(test_loader):
            #     encoding = {k: v.to(device=self.config.device) for k, v in encoding.items()}
            #     prediction = model(encoding)
            #     if self.config.loss == 'CrossEntropy':
            #         predicted_label = self.softmax(prediction)
            #     else:
            #         predicted_label = prediction
            #     predicted_label = torch.argmax(predicted_label, dim=1)
            #     preds += predicted_label.cpu()
            #     labels += label.cpu()
            # print("Test Accuracy: {}".format(np.sum(np.array(labels) == np.array(preds)) / len(preds)))

