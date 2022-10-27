from transformers import RobertaForMultipleChoice
import torch.nn as nn
import torch


class Roberta(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = RobertaForMultipleChoice.from_pretrained(
            'LIAMF-USP/roberta-large-finetuned-race')
        self.linear = nn.Linear(2, 2)
        self.softmax = nn.Softmax()

    def forward(self, encoding, label):
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        roberta_output = self.model(**{k: v for k, v in encoding.items()}, labels=label)
        loss = roberta_output.loss
        roberta_output = roberta_output.logits
        if self.config.loss == 'CrossEntropy':
            output = roberta_output
        else:
            output = self.softmax(roberta_output)
        return output, loss
