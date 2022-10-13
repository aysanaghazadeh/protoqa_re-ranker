from torch.utils.data import Dataset


class Dataset(Dataset):

    def __init__(self, data):
        self.answers = data['answers']
        self.questions = data['questions']

    def __len__(self):
        return len(self.answers)

    def __getitem__(self, idx):
        question = self.questions[idx]
        first_answer = self.answers[idx]['first_answer']
        second_answer = self.answers[idx]['second_answer']
        return question, first_answer, second_answer

