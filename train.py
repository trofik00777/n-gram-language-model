from collections import defaultdict
import pickle
import string
import numpy as np
import argparse
import sys
import os


class Model:
    def __init__(self, model_path: str = None):
        if model_path is not None:
            self.data = self.load(model_path)
        else:
            self.data = defaultdict(dict)

    def fit(self, text: str):
        preproc_tokens = list(
            filter(lambda x: x.isalpha(), text.strip().lower().translate(
                str.maketrans(string.punctuation, ' ' * len(string.punctuation))
            ).split())
        )

        # сейчас префикс из 1 эл-та, тк думаю если для нескольких, то стоит приводить слова
        # в начальную форму, чтобы лучше хранились, но сейчас использовать специальные библиотеки нельзя

        for i_word in range(len(preproc_tokens) - 1):
            if preproc_tokens[i_word + 1] in self.data[preproc_tokens[i_word]]:
                self.data[preproc_tokens[i_word]][preproc_tokens[i_word + 1]] += 1
            else:
                self.data[preproc_tokens[i_word]][preproc_tokens[i_word + 1]] = 1

    def generate(self, length: int, first: str = None):
        keys = list(self.data.keys())

        if first is None:
            first = np.random.choice(keys)

        generated = [first]  # type: list[str]
        for _ in range(length):
            words, counts = list(zip(*self.data.get(generated[-1], {np.random.choice(keys): 1}).items()))
            s_counts = sum(counts)
            p = [x / s_counts for x in counts]
            predict = np.random.choice(
                words,
                p=p
            )
            generated.append(predict)

        return ' '.join(generated)

    def load(self, path: str):
        return pickle.load(open(path, 'rb'))

    def save(self, path: str):
        pickle.dump(self.data, open(path, 'wb'))


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-dir", type=str, dest='input_dir')
    parser.add_argument("--model", required=True, type=str)

    return parser.parse_args()


def main():
    args = _parse_args()

    model = Model()
    if args.input_dir is None:
        for line in sys.argv:
            model.fit(line)
    else:
        for filename in os.listdir(args.input_dir):
            model.fit(open(f"{args.input_dir.rstrip('/')}/{filename}", 'r', encoding='utf-8').read())

    model.save(args.model)


if __name__ == "__main__":
    main()
