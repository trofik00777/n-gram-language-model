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
            self.data = defaultdict(list)

    def fit(self, text: str):
        preproc_tokens = list(
            filter(lambda x: x.isalpha(), text.strip().lower().translate(
                str.maketrans('', '', string.punctuation)
            ).split())
        )
        for i_word in range(len(preproc_tokens) - 1):
            self.data[preproc_tokens[i_word]].append(preproc_tokens[i_word + 1])

    def generate(self, length: int, first: str = None):
        keys = list(self.data.keys())

        if first is None:
            first = np.random.choice(keys)

        generated = [first]  # type: list[str]
        for _ in range(length):
            print(generated[-1])
            predict = np.random.choice(self.data.get(generated[-1], [np.random.choice(keys)]))
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
            model.fit(open(f"{args.input_dir}/{filename}", 'r').read())

    model.save(args.model)


if __name__ == "__main__":
    main()
