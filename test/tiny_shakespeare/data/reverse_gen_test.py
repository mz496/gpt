import itertools
import random
from pathlib import Path

import pytest

from src.tiny_shakespeare.data.sft import reverse_gen
from src.tiny_shakespeare.data.sft.reverse_gen import ALPHABET, CATEGORIES, FORCED, PER_SLICE, classify, generate_slices, lengths_for, make, spread
from src.tiny_shakespeare.data.sft.reverse_source import SLICES_DIRNAME, TRAIN_FILENAME

SFT_DIR = Path(reverse_gen.__file__).parent


def lines(path: Path) -> list[tuple[str, str]]:
    return [(a, b) for a, b in (line.split("\t") for line in path.read_text().splitlines())]


# --- the categories

@pytest.mark.parametrize("string, category", [
    ("aaaaa", "repeated_char"),
    ("abcba", "palindrome"),
    ("ababab", "repeated_seq"),
    ("abcabca", "repeated_seq"),  # a unit repeated and cut short
    ("abcdef", "random"),
    ("a", "random"),  # length 1 is trivially all of them, so it counts as random
    ("aa", "repeated_char"),  # a palindrome too, but repeated_char comes first
    ("ababa", "palindrome"),  # a 2-char unit repeated at length 5 is always a palindrome
])
def test_classify(string: str, category: str) -> None:
    assert classify(string) == category


@pytest.mark.parametrize("category", CATEGORIES)
def test_make_gives_strings_with_the_property_of_its_category(category: str) -> None:
    rng = random.Random(0)
    for n in lengths_for(category):
        s = make(rng, category, n)
        assert len(s) == n and set(s) <= set(ALPHABET)
        if category == "repeated_char":
            assert len(set(s)) == 1
        elif category == "palindrome":
            assert s == s[::-1]
        elif category == "repeated_seq":
            assert reverse_gen.is_repeated_seq(s)


# --- the train file

def test_the_committed_train_file_is_well_formed() -> None:
    examples = lines(SFT_DIR / TRAIN_FILENAME)
    strings = [a for a, _ in examples]
    assert all(b == a[::-1] and a.isalpha() and a.islower() and 1 <= len(a) <= reverse_gen.MAX_LEN for a, b in examples)
    assert len(set(strings)) == len(strings)
    assert [len(s) for s in strings] == sorted(len(s) for s in strings)
    assert set(FORCED) <= set(strings)
    assert len(strings) == 4926


# --- the files as a whole

def test_regenerating_everything_gives_the_committed_files_byte_for_byte(tmp_path: Path) -> None:
    # The checkpoints were trained on the train file, and the slices are clean for them only while the two stay as they are,
    # so a change in how either is generated must not slip through. Changing it on purpose means regenerating and committing the files.
    reverse_gen.main(tmp_path)
    names = [TRAIN_FILENAME] + [f"{SLICES_DIRNAME}/{category}.txt" for category in CATEGORIES]
    for name in names:
        assert (tmp_path / name).read_bytes() == (SFT_DIR / name).read_bytes(), name


# --- the validation slices

def test_spread_splits_a_total_evenly_and_adds_up() -> None:
    parts = spread(300, list(range(3, 51)))
    assert sum(parts.values()) == 300
    assert max(parts.values()) - min(parts.values()) <= 1
    # Cumulative rounding of 1.25, 2.5, 3.75, 5: the one extra string lands in the middle, not at the start.
    assert spread(5, [1, 2, 3, 4]) == {1: 1, 2: 1, 3: 2, 4: 1}


def test_the_validation_slices_have_one_file_per_built_in_category() -> None:
    assert sorted(p.name for p in (SFT_DIR / SLICES_DIRNAME).iterdir()) == sorted(f"{category}.txt" for category in CATEGORIES)


@pytest.mark.parametrize("category", CATEGORIES)
def test_each_committed_slice_holds_300_correct_strings_of_its_own_category(category: str) -> None:
    examples = lines(SFT_DIR / SLICES_DIRNAME / f"{category}.txt")
    strings = [a for a, _ in examples]
    assert len(strings) == PER_SLICE
    assert all(b == a[::-1] for a, b in examples)
    assert [len(s) for s in strings] == sorted(len(s) for s in strings)
    assert all(classify(s) == category for s in strings), "the file a string is in must be its category"


def test_the_committed_slices_share_no_string_with_the_train_file() -> None:
    sliced = [a for category in CATEGORIES for a, _ in lines(SFT_DIR / SLICES_DIRNAME / f"{category}.txt")]
    trained_on = {a for a, _ in lines(SFT_DIR / TRAIN_FILENAME)}
    assert len(set(sliced)) == len(sliced), "no string in two files, or twice in one"
    assert not set(sliced) & trained_on, "the validation slices must share no string with what the model trained on"


def test_a_length_with_too_few_unused_strings_has_its_shortfall_made_up_elsewhere() -> None:
    # Every string of length 1 and 2 is taken, so the random category can't use those lengths, but still gets all of its strings.
    taken = {"".join(p) for n in (1, 2) for p in itertools.product(ALPHABET, repeat=n)}
    by_category, missing = generate_slices(taken, random.Random(0))
    assert missing == {category: 0 for category in CATEGORIES}
    for category in CATEGORIES:
        assert len(by_category[category]) == PER_SLICE
        assert all(len(s) >= 3 for s in by_category[category])
        assert not set(by_category[category]) & taken
