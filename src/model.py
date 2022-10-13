from transformers import RobertaTokenizer, RobertaForMultipleChoice
import torch.nn as nn
import torch


class Roberta(nn.Module):
    def __init__(self, config):
        self.config = config
        self.tokenizer = RobertaTokenizer.from_pretrained('bert-base')
        self.model = RobertaForMultipleChoice
        self.linear = nn.Linear(config.num_choices, config.num_choices)


    def forward(self, question, choices):
        labels = torch.tensor(0).unsqueeze(0)
        encoding = self.tokenizer([question] * len(choices), choices, return_tensors="pt", padding=True)
        roberta_output = self.model(**{k: v.unsqueeze(0) for k, v in encoding.items()}, labels=labels)
        linear_output = self.linear(roberta_output['logits'])
        return nn.Softmax(linear_output)



