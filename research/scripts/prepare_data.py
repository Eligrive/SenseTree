"""Downloads the benchmark datasets and writes them in the BEIR layout under data/."""

import argparse

from sensetree_research.data import DATASETS, load_corpus, load_qrels, has_split, prepare


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", default=list(DATASETS))
    args = parser.parse_args()
    for name in args.names:
        prepare(name)
        corpus = load_corpus(name)
        sizes = {s: len(load_qrels(name, s)) for s in ("train", "dev", "test") if has_split(name, s)}
        print(f"{name:10s} docs={len(corpus):6d} " + " ".join(f"{s}={n}" for s, n in sizes.items()))


if __name__ == "__main__":
    main()
