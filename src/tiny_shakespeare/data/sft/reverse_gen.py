"""
Generates the data of the string reversal task next to this file, one `string<TAB>reversed` per line:
- reverse-train.txt: stratified by (category, length).
- reverse-validation-slices/<category>.txt: the validation data, sliced by subtask (the category of the string), one file per slice. They are what the loss and the
  per-subtask scores are measured on while a model trains, so each built-in slice has the same number of strings (PER_SLICE), spread evenly over the lengths it
  can have. None of them is in the train file. A slice of your own is just another file there, and should keep to that last rule.
Also the home of the categories (see classify), which the scoring of the slices shares.

Run, from the repo root: uv run python -m src.tiny_shakespeare.data.sft.reverse_gen
Rerunning it must give the files that are in the repo byte for byte (see reverse_gen_test.py): the checkpoints were trained on the train file, and the slices
are clean for them only while the two stay as they are.
"""
import random
from pathlib import Path

from src.tiny_shakespeare.data.sft.reverse_source import SLICES_DIRNAME, TRAIN_FILENAME

SEED = 1337
ALPHABET = "abcdefghijklmnopqrstuvwxyz"
MAX_LEN = 50
PER_LENGTH = 100  # strings per length in the train file (length 1 only has 26 possible strings)
# Share of each length's strings; "random" is the remainder (and absorbs lengths where a category can't exist).
SHARES = {"palindrome": 0.12, "repeated_char": 0.08, "repeated_seq": 0.12}
MIN_LEN = {"random": 1, "palindrome": 3, "repeated_char": 2, "repeated_seq": 4}
# Length 5 can only repeat a 2-char unit (ababa), which is always a palindrome, so it has no repeated_seq.
NO_REPEATED_SEQ = {5}
# Always in the train file.
FORCED = ["abcba", "ababab", "aaaaa"]
OUT_DIR = Path(__file__).parent

SLICES_SEED = 2718  # not the seed of the train generator, so the streams of random numbers are unrelated
PER_SLICE = 300
# Give up on a length after this many tries in a row that found no new string (e.g. length 1 has 26 strings in all, and the train file uses them).
MAX_FAILED_ATTEMPTS = 5000

CATEGORIES = ["palindrome", "repeated_char", "repeated_seq", "random"]

def rand(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(ALPHABET) for _ in range(n))

def is_repeated_seq(s: str) -> bool:
    n = len(s)
    return any(n >= 2 * u and s == (s[:u] * n)[:n] for u in range(2, 7))

def classify(s: str) -> str:
    """By property, first match wins. Length 1 is trivially all three, so it counts as random."""
    if len(s) == 1:
        return "random"
    if len(set(s)) == 1:
        return "repeated_char"
    if s == s[::-1]:
        return "palindrome"
    if is_repeated_seq(s):
        return "repeated_seq"
    return "random"

def lengths_for(cat: str) -> list[int]:
    """The lengths at which a string of this category can exist."""
    return [n for n in range(MIN_LEN[cat], MAX_LEN + 1) if not (cat == "repeated_seq" and n in NO_REPEATED_SEQ)]

def make(rng: random.Random, cat: str, n: int) -> str:
    """A random string of length n that is meant to be of the category (the caller checks with classify, since e.g. a random one can come out as a palindrome)."""
    if cat == "random":
        return rand(rng, n)
    if cat == "repeated_char":
        return rng.choice(ALPHABET) * n
    if cat == "palindrome":
        half = rand(rng, n // 2)
        return half + (rand(rng, 1) if n % 2 else "") + half[::-1]
    unit = rand(rng, rng.randint(2, min(6, n // 2)))
    return (unit * n)[:n]

def write_examples(path: Path, strings: set[str]) -> None:
    """One `string<TAB>reversed` line per string, shortest first."""
    with open(path, "w") as f:
        for s in sorted(strings, key=lambda s: (len(s), s)):
            f.write(f"{s}\t{s[::-1]}\n")

def generate_train(rng: random.Random) -> set[str]:
    """The strings of the train file: PER_LENGTH per length, split between the categories by SHARES."""
    # Strings per (category, length).
    counts: dict[str, dict[int, int]] = {c: {} for c in CATEGORIES}
    for n in range(1, MAX_LEN + 1):
        total = 26 if n == 1 else PER_LENGTH
        for c in CATEGORIES[:3]:
            counts[c][n] = round(SHARES[c] * total) if n >= MIN_LEN[c] and not (c == "repeated_seq" and n in NO_REPEATED_SEQ) else 0
        counts["random"][n] = total - sum(counts[c][n] for c in CATEGORIES[:3])

    seen: set[str] = set()
    cells: dict[tuple[str, int], list[str]] = {}
    for n in range(1, MAX_LEN + 1):
        for c in CATEGORIES:
            items = [s for s in FORCED if (classify(s), len(s)) == (c, n)]
            seen.update(items)
            attempts = 0
            while len(items) < counts[c][n]:
                attempts += 1
                assert attempts < 100_000, f"cannot make {counts[c][n]} {c} strings of length {n}"
                s = make(rng, c, n)
                if s not in seen and classify(s) == c:
                    seen.add(s)
                    items.append(s)
            cells[(c, n)] = items
    return {s for items in cells.values() for s in items}

def spread(total: int, keys: list[int]) -> dict[int, int]:
    """Splits `total` over the keys as evenly as possible, with the extra ones spread out, not all at the start."""
    return {key: round((i + 1) * total / len(keys)) - round(i * total / len(keys)) for i, key in enumerate(keys)}

def generate_slices(exclude: set[str], rng: random.Random) -> tuple[dict[str, list[str]], dict[str, int]]:
    """
    PER_SLICE new strings per category, none of them in `exclude`. Returns them by category, and how many are missing from each category (0 if complete).
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
        for n, quota in spread(PER_SLICE, lengths).items():
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
    train = generate_train(random.Random(SEED))
    # The slices are made after the train file, from their own random numbers, so that they can't change it, and they avoid every string in it.
    slices, missing = generate_slices(train, random.Random(SLICES_SEED))

    write_examples(out_dir / TRAIN_FILENAME, train)
    print(f"train          {len(train):4d} strings")
    slices_dir = out_dir / SLICES_DIRNAME
    slices_dir.mkdir(exist_ok=True)
    for cat in CATEGORIES:
        write_examples(slices_dir / f"{cat}.txt", set(slices[cat]))
        lengths = sorted(len(s) for s in slices[cat])
        print(f"{cat:14s} {len(lengths):4d} strings, lengths {lengths[0]}-{lengths[-1]}" + (f", MISSING {missing[cat]}" if missing[cat] else ""))

if __name__ == "__main__":
    main()
