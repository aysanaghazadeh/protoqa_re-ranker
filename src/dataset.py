from torch.utils.data import Dataset
from random import shuffle


class Dataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.answers = data['answers']
        self.questions = data['questions']

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

        return question, choices, label
