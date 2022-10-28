from torch.utils.data import Dataset
from random import shuffle
from transformers import RobertaTokenizer
import torch


class Dataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.answers = data['answers']
        self.questions = data['questions']
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    def __len__(self):
        return len(self.answers)

    def __getitem__(self, idx):
        question = self.questions[idx]
        first_answer = self.answers[idx]['first_answer']
        second_answer = self.answers[idx]['second_answer']
        choices = [first_answer, second_answer]
        shuffle(choices)
        if choices[0] == first_answer:
            label = 0
        else:
            label = 1
        encoding = self.tokenizer([question, question], choices, return_tensors="pt", padding='max_length',
                                  truncation=True, max_length=50)
        encoding = {key: val for key, val in encoding.items()}
        return encoding, label
