"""
Generates reverse-train.txt and reverse-validation.txt next to this file: stratified by (category, length), disjoint, one `string<TAB>reversed` per line.

Run: python3 src/tiny_shakespeare/data/finetuning/reverse_gen.py
"""
import random
from pathlib import Path

SEED = 1337
ALPHABET = "abcdefghijklmnopqrstuvwxyz"
MAX_LEN = 50
PER_LENGTH = 100  # strings per length across both splits (length 1 only has 26 possible strings)
VAL_FRAC = 0.1
# Share of each length's strings; "random" is the remainder (and absorbs lengths where a category can't exist).
SHARES = {"palindrome": 0.12, "repeated_char": 0.08, "repeated_seq": 0.12}
MIN_LEN = {"random": 1, "palindrome": 3, "repeated_char": 2, "repeated_seq": 4}
# Length 5 can only repeat a 2-char unit (ababa), which is always a palindrome, so it has no repeated_seq.
NO_REPEATED_SEQ = {5}
# Always in the train split.
FORCED = ["abcba", "ababab", "aaaaa"]
OUT_DIR = Path(__file__).parent

rng = random.Random(SEED)

def rand(n: int) -> str:
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

def make(cat: str, n: int) -> str:
    if cat == "random":
        return rand(n)
    if cat == "repeated_char":
        return rng.choice(ALPHABET) * n
    if cat == "palindrome":
        half = rand(n // 2)
        return half + (rand(1) if n % 2 else "") + half[::-1]
    unit = rand(rng.randint(2, min(6, n // 2)))
    return (unit * n)[:n]

CATS = ["palindrome", "repeated_char", "repeated_seq", "random"]

# Strings per (category, length).
counts: dict[str, dict[int, int]] = {c: {} for c in CATS}
for n in range(1, MAX_LEN + 1):
    total = 26 if n == 1 else PER_LENGTH
    for c in CATS[:3]:
        counts[c][n] = round(SHARES[c] * total) if n >= MIN_LEN[c] and not (c == "repeated_seq" and n in NO_REPEATED_SEQ) else 0
    counts["random"][n] = total - sum(counts[c][n] for c in CATS[:3])

seen: set[str] = set()
cells: dict[tuple[str, int], list[str]] = {}
for n in range(1, MAX_LEN + 1):
    for c in CATS:
        items = [s for s in FORCED if (classify(s), len(s)) == (c, n)]
        seen.update(items)
        attempts = 0
        while len(items) < counts[c][n]:
            attempts += 1
            assert attempts < 100_000, f"cannot make {counts[c][n]} {c} strings of length {n}"
            s = make(c, n)
            if s not in seen and classify(s) == c:
                seen.add(s)
                items.append(s)
        cells[(c, n)] = items

# Validation: for each category, ~VAL_FRAC of each length's strings, so lengths and categories match the train split.
val: set[str] = set()
for c in CATS:
    ts = {n: counts[c][n] for n in counts[c] if counts[c][n]}
    cat_total = sum(ts.values())
    val_total = round(VAL_FRAC * cat_total)
    quotas = {n: val_total * t / cat_total for n, t in ts.items()}
    alloc = {n: int(q) for n, q in quotas.items()}
    leftovers = sorted(ts, key=lambda n: (-(quotas[n] - alloc[n]), rng.random()))
    for n in leftovers[: val_total - sum(alloc.values())]:
        alloc[n] += 1
    for n, k in alloc.items():
        candidates = [s for s in cells[(c, n)] if s not in FORCED]
        val.update(rng.sample(candidates, k))

all_strings = {s for items in cells.values() for s in items}
train = all_strings - val

def write(name: str, strings: set[str]) -> None:
    with open(OUT_DIR / name, "w") as f:
        for s in sorted(strings, key=lambda s: (len(s), s)):
            f.write(f"{s}\t{s[::-1]}\n")

write("reverse-train.txt", train)
write("reverse-validation.txt", val)
