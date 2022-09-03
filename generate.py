import argparse
from fit import Model


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--prefix", type=str, nargs='+')
    parser.add_argument("--length", required=True, type=int)
    parser.add_argument("--model", required=True, type=str)

    return parser.parse_args()


def main():
    args = _parse_args()

    model = Model(args.model)
    if args.prefix is not None:
        generated_text = ' '.join(args.prefix[:-1]) + " " + model.generate(args.length, args.prefix[-1])
    else:
        generated_text = model.generate(args.length)

    print(generated_text)


if __name__ == "__main__":
    main()
