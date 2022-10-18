from transformers import RobertaTokenizer, RobertaForMultipleChoice
import torch.nn as nn
import torch


class Roberta(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.model = RobertaForMultipleChoice.from_pretrained('roberta-base')
        self.linear = nn.Linear(config.num_choices, 1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, encoding):
        labels = torch.tensor(0).unsqueeze(0).to(device=self.config.device)
        roberta_output = self.model(**{k: v.unsqueeze(0) for k, v in encoding.items()}, labels=labels)
        # linear_output = self.linear(roberta_output['logits'])
        return roberta_output['logits']
