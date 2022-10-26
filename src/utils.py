import json
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataset import Dataset
import os
import warnings
from itertools import product


def load_question_answer_list(config):
    data_path = config.path_to_dataset
    questions = []
    answers = []
    with open(data_path) as data:
        for q in data:
            q_json = json.loads(q)
            if "answers" in q_json and "clusters" in q_json["answers"]:
                question = q_json["question"]["normalized"]
                answer_list = []
                answer_ranks = []
                for cluster_id in q_json["answers"]["clusters"]:
                    cluster = q_json["answers"]["clusters"][cluster_id]
                    answer_rank = int(cluster_id.split('.')[-1])
                    for answer in cluster["answers"]:
                        answer_list.append(answer)
                        answer_ranks.append(answer_rank)
                for (index_one, answer_one), (index_two, answer_two) in product(enumerate(answer_list), enumerate(answer_list)):
                    if answer_ranks[index_one] == answer_ranks[index_two]:
                        continue
                    else:
                        ans = {}
                        if answer_ranks[index_one] < answer_ranks[index_two]:
                            ans["first_answer"] = answer_one
                            ans["second_answer"] = answer_two
                        else:
                            ans["first_answer"] = answer_two
                            ans["second_answer"] = answer_one
                        questions.append(question)
                        answers.append(ans)
            else:
                warnings.warn(
                    f"Data in {data_path} seems to be using an old format. "
                    f"We will attempt to load anyway, but you should download the newest version."
                )
                if "answers-cleaned" in q_json and isinstance(
                        q_json["answers-cleaned"], list
                ):
                    answer_clusters = {
                        frozenset(ans_cluster["answers"]): ans_cluster["count"]
                        for ans_cluster in q_json["answers-cleaned"]
                    }
                elif "answers-cleaned" in q_json and isinstance(
                        q_json["answers-cleaned"], dict
                ):
                    answer_clusters = {
                        frozenset([answer]): count
                        for answer, count in q_json["answers-cleaned"].items()
                    }
                else:
                    raise ValueError(
                        f"Could not load data from {data_path}, unable to find answer clusters."
                    )

    return answers, questions


def load_data(config):
    answers, questions = load_question_answer_list(config)
    answers, questions = answers[0:13], questions[0:13]
    train_questions, test_questions, train_answers, test_answers = train_test_split(
        questions,
        answers,
        test_size=config.test_size,
        random_state=7)
    data = {"questions": train_questions, "answers": train_answers}
    train_set = Dataset(data)
    data = {"questions": test_questions, "answers": test_answers}
    test_set = Dataset(data)
    print(f"[INFO] found {len(train_set)} examples in the training set...")
    print(f"[INFO] found {len(test_set)} examples in the test set...")
    train_loader = DataLoader(train_set, shuffle=False, batch_size=config.batch_size, num_workers=os.cpu_count())
    test_loader = DataLoader(test_set, shuffle=False, batch_size=config.batch_size, num_workers=os.cpu_count())
    return train_loader, test_loader
