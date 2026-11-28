"""
Generates the eval set in the reverse-eval directory next to this file: larger than the validation split, to measure how well a finetuned model does on each
category of string. There is one file per category (reverse-eval/<category>.txt), so the category of a string is the file it is in, and a category of your own
is just another file there. One `string<TAB>reversed` per line.
Each built-in category has the same number of strings (PER_CATEGORY), spread evenly over the lengths it can have, and none of them is in
reverse-train.txt or reverse-validation.txt, so the set is clean for any checkpoint trained on those. Any file you add should keep to that last rule.

Run, from the repo root: uv run python -m src.tiny_shakespeare.data.sft.reverse_eval_gen
Needs the train and validation files to exist, and to be the ones the checkpoints were trained on (don't regenerate those with other settings first).
"""
import random
from pathlib import Path

from src.tiny_shakespeare.data.sft.reverse_gen import CATEGORIES, classify, lengths_for, make, write_examples
from src.tiny_shakespeare.data.sft.reverse_source import EVAL_DIRNAME, ReverseSource

EVAL_SEED = 2718  # not the seed of the train/validation generator, so the streams of random numbers are unrelated
PER_CATEGORY = 300
# Give up on a length after this many tries in a row that found no new string (e.g. length 1 has 26 strings in all, and the train/validation files use them).
MAX_FAILED_ATTEMPTS = 5000
OUT_DIR = Path(__file__).parent

def used_strings() -> set[str]:
    """Every string in the train and validation files."""
    source = ReverseSource()
    return {example.input for split in ("train", "validation") for example in source.iter_examples(split)}

def spread(total: int, keys: list[int]) -> dict[int, int]:
    """Splits `total` over the keys as evenly as possible, with the extra ones spread out, not all at the start."""
    return {key: round((i + 1) * total / len(keys)) - round(i * total / len(keys)) for i, key in enumerate(keys)}

def generate(exclude: set[str], rng: random.Random) -> tuple[dict[str, list[str]], dict[str, int]]:
    """
    PER_CATEGORY new strings per category, none of them in `exclude`. Returns them by category, and how many are missing from each category (0 if complete).
    Where a length has too few unused strings, its shortfall is made up at the other lengths.
    """
    chosen = set(exclude)
    result: dict[str, list[str]] = {}
    missing: dict[str, int] = {}
    for cat in CATEGORIES:
        strings: list[str] = []

        def add(n: int, how_many: int) -> int:
            """Tries to add `how_many` new strings of length n; returns how many it did add."""
            added, failed = 0, 0
            while added < how_many and failed < MAX_FAILED_ATTEMPTS:
                s = make(rng, cat, n)
                if s in chosen or classify(s) != cat:
                    failed += 1
                    continue
                chosen.add(s)
                strings.append(s)
                added, failed = added + 1, 0
            return added

        lengths = lengths_for(cat)
        carry = 0
        for n, quota in spread(PER_CATEGORY, lengths).items():
            carry = quota + carry - add(n, quota + carry)
        # Whatever the lengths couldn't give is taken from the longest ones, which have the most strings to choose from.
        for n in reversed(lengths):
            if carry == 0:
                break
            carry -= add(n, carry)
        result[cat] = strings
        missing[cat] = carry
    return result, missing

def main(out_dir: Path = OUT_DIR) -> None:
    by_category, missing = generate(used_strings(), random.Random(EVAL_SEED))
    eval_dir = out_dir / EVAL_DIRNAME
    eval_dir.mkdir(exist_ok=True)
    for cat in CATEGORIES:
        write_examples(eval_dir / f"{cat}.txt", set(by_category[cat]))
        lengths = sorted(len(s) for s in by_category[cat])
        print(f"{cat:14s} {len(lengths):4d} strings, lengths {lengths[0]}-{lengths[-1]}" + (f", MISSING {missing[cat]}" if missing[cat] else ""))

if __name__ == "__main__":
    main()
