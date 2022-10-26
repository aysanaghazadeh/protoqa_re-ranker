from transformers import RobertaTokenizer, RobertaForMultipleChoice
import torch.nn as nn
import torch


class Roberta(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = RobertaForMultipleChoice.from_pretrained(
            'roberta-base')
        self.linear = nn.Linear(2, 2)
        self.softmax = nn.Softmax()

    def forward(self, encoding):
        roberta_output = self.model(**{k: v.unsqueeze(0) for k, v in encoding.items()})
        roberta_output = torch.transpose(roberta_output.logits, 0, 1)
        if self.config.loss == 'CrossEntropy':
            output = roberta_output
        else:
            output = self.softmax(roberta_output)
        return output
