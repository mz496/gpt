import json
from pathlib import Path

import pytest

from src.tiny_shakespeare.tokenizers.bpe_tokenizer import BPETokenizer
from src.tiny_shakespeare.tokenizers.char_tokenizer import CharTokenizer
from src.tiny_shakespeare.tokenizers.regex_bpe_tokenizer import RegexBPETokenizer
from src.tiny_shakespeare.tokenizers.tokenizer import Tokenizer
from src.tiny_shakespeare.tokenizers.utils import load_tokenizer, make_tokenizer

EOT = "<|endoftext|>"
PAD = "<|pad|>"
TEXT = "the quick brown fox jumps over the lazy dog, then the dog chases the fox. " * 20
BPE_NAMES = ["bpe", "regex_bpe"]


def trained(name: str, special_tokens: list[str], vocab_size: int = 290, text: str = TEXT) -> Tokenizer:
    tokenizer = make_tokenizer(name, vocab_size)
    tokenizer.register_special_tokens(special_tokens)
    tokenizer.train(text)
    return tokenizer


# --- registration

def test_registering_replaces_the_previous_special_tokens() -> None:
    tokenizer = CharTokenizer()
    tokenizer.register_special_tokens([EOT, PAD])
    tokenizer.register_special_tokens([PAD])
    assert tokenizer.special_tokens == [PAD]


def test_registering_an_empty_list_clears_the_special_tokens() -> None:
    tokenizer = CharTokenizer()
    tokenizer.register_special_tokens([EOT])
    tokenizer.register_special_tokens([])
    tokenizer.train("abc" + EOT)
    assert tokenizer.special_tokens == []
    assert EOT not in tokenizer.get_vocab().values()
    assert tokenizer.decode_as_string(tokenizer.encode_string("abc" + EOT)) == "abc" + EOT  # now just ordinary characters


def test_special_tokens_returns_a_copy() -> None:
    tokenizer = CharTokenizer()
    tokenizer.register_special_tokens([EOT])
    tokenizer.special_tokens.append("mutated")
    assert tokenizer.special_tokens == [EOT]


@pytest.mark.parametrize("bad", [[""], [EOT, EOT]])
def test_empty_and_duplicate_special_tokens_raise(bad: list[str]) -> None:
    with pytest.raises(ValueError):
        CharTokenizer().register_special_tokens(bad)


@pytest.mark.parametrize("name", ["char", *BPE_NAMES])
def test_registering_after_train_raises(name: str) -> None:
    tokenizer = trained(name, [])
    with pytest.raises(ValueError, match="before train"):
        tokenizer.register_special_tokens([EOT])


# --- the vocab and merge budget

@pytest.mark.parametrize("name", BPE_NAMES)
@pytest.mark.parametrize("specials", [[], [EOT], [EOT, PAD]])
def test_the_vocab_size_is_the_target_including_the_special_tokens(name: str, specials: list[str]) -> None:
    tokenizer = trained(name, specials, vocab_size=290)
    assert len(tokenizer.get_vocab()) == 290


@pytest.mark.parametrize("name", BPE_NAMES)
def test_a_target_with_no_room_for_merges_raises(name: str) -> None:
    tokenizer = make_tokenizer(name, 258)
    tokenizer.register_special_tokens([EOT, PAD])
    with pytest.raises(ValueError, match="no room for merges"):
        tokenizer.train(TEXT)


@pytest.mark.parametrize("name", BPE_NAMES)
def test_special_tokens_only_take_merges_off_the_end(name: str) -> None:
    without = trained(name, [])
    with_two = trained(name, [EOT, PAD])

    assert isinstance(without, (BPETokenizer, RegexBPETokenizer)) and isinstance(with_two, (BPETokenizer, RegexBPETokenizer))
    assert len(with_two.merges) == len(without.merges) - 2
    assert with_two.merges == without.merges[:len(with_two.merges)]


def test_char_vocab_is_the_characters_then_the_special_tokens() -> None:
    tokenizer = CharTokenizer()
    tokenizer.register_special_tokens([EOT, PAD])
    tokenizer.train("cab")
    assert tokenizer.get_vocab() == {0: b"a", 1: b"b", 2: b"c", 3: EOT.encode(), 4: PAD.encode()}


@pytest.mark.parametrize("name", BPE_NAMES)
def test_special_token_ids_come_after_all_the_merges_in_registration_order(name: str) -> None:
    tokenizer = trained(name, [PAD, EOT])
    assert isinstance(tokenizer, (BPETokenizer, RegexBPETokenizer))
    first_special = 256 + len(tokenizer.merges)

    assert tokenizer.special_token_ids == {PAD: first_special, EOT: first_special + 1}
    assert tokenizer.get_vocab()[first_special] == PAD.encode()
    assert tokenizer.get_vocab()[first_special + 1] == EOT.encode()
    assert all(max(*merge.from_tokens, merge.to_token) < first_special for merge in tokenizer.merges)


def test_the_learned_ids_dont_depend_on_the_special_tokens_for_char() -> None:
    plain = CharTokenizer()
    plain.train("hello world")
    special = CharTokenizer()
    special.register_special_tokens([EOT])
    special.train("hello world")
    assert special.encode_string("hello world") == plain.encode_string("hello world")


# --- training

def test_the_char_vocab_gets_nothing_from_the_special_text() -> None:
    tokenizer = CharTokenizer()
    tokenizer.register_special_tokens([EOT])
    tokenizer.train("ab" + EOT + "ba")
    assert tokenizer.vocab == ["a", "b"]


def test_without_registering_the_special_text_is_learned_like_any_other_text() -> None:
    tokenizer = CharTokenizer()
    tokenizer.train("ab" + EOT + "ba")
    assert set(EOT) <= set(tokenizer.vocab)


@pytest.mark.parametrize("name", BPE_NAMES)
def test_no_merge_is_learned_from_or_across_a_special_token(name: str) -> None:
    # With the special tokens removed, every piece is a single letter, so there is no adjacent pair to merge.
    # Every adjacent pair in the raw text either contains part of the special token or crosses it.
    tokenizer = trained(name, [EOT], text=("a" + EOT + "b" + EOT) * 20)

    assert isinstance(tokenizer, (BPETokenizer, RegexBPETokenizer))
    assert tokenizer.merges == []


@pytest.mark.parametrize("name", BPE_NAMES)
def test_the_special_text_is_merged_like_any_other_text_when_not_registered(name: str) -> None:
    tokenizer = make_tokenizer(name, 290)
    tokenizer.train(("a" + EOT + "b" + EOT) * 20)
    assert isinstance(tokenizer, (BPETokenizer, RegexBPETokenizer))
    assert len(tokenizer.merges) > 0


# --- encode and decode

@pytest.mark.parametrize("name", ["char", *BPE_NAMES])
def test_encode_puts_the_special_id_where_the_special_token_was(name: str) -> None:
    tokenizer = trained(name, [EOT])
    eot = tokenizer.special_token_ids[EOT]

    assert tokenizer.encode_string("the fox" + EOT + "the dog") == tokenizer.encode_string("the fox") + [eot] + tokenizer.encode_string("the dog")


@pytest.mark.parametrize("name", ["char", *BPE_NAMES])
def test_leading_trailing_and_consecutive_special_tokens(name: str) -> None:
    tokenizer = trained(name, [EOT])
    eot = tokenizer.special_token_ids[EOT]
    x = tokenizer.encode_string("the")

    assert tokenizer.encode_string(EOT + "the" + EOT + EOT) == [eot] + x + [eot, eot]
    assert tokenizer.encode_string(EOT) == [eot]
    assert tokenizer.encode_string("") == []


@pytest.mark.parametrize("name", ["char", *BPE_NAMES])
def test_text_without_special_tokens_encodes_the_same_with_or_without_registering(name: str) -> None:
    assert trained(name, [EOT], vocab_size=290).encode_string("the lazy dog") == trained(name, [], vocab_size=289).encode_string("the lazy dog")


@pytest.mark.parametrize("name", ["char", *BPE_NAMES])
def test_decode_round_trips_text_with_special_tokens(name: str) -> None:
    tokenizer = trained(name, [EOT, PAD])
    text = "the fox" + EOT + "the dog" + PAD + EOT + EOT + "chases"
    assert tokenizer.decode_as_string(tokenizer.encode_string(text)) == text


@pytest.mark.parametrize("name", ["char", *BPE_NAMES])
def test_decode_turns_a_special_id_into_its_text(name: str) -> None:
    tokenizer = trained(name, [EOT])
    eot = tokenizer.special_token_ids[EOT]
    assert tokenizer.decode_as_string([eot]) == EOT
    assert tokenizer.decode_as_string(tokenizer.encode_string("the") + [eot]) == "the" + EOT


@pytest.mark.parametrize("name", ["char", *BPE_NAMES])
def test_the_longest_special_token_wins_when_one_is_a_prefix_of_another(name: str) -> None:
    tokenizer = trained(name, ["<a>", "<a><b>"])
    short, long = tokenizer.special_token_ids["<a>"], tokenizer.special_token_ids["<a><b>"]

    assert tokenizer.encode_string("<a><b>") == [long]
    assert tokenizer.encode_string("<a>") == [short]
    assert tokenizer.encode_string("<a><a>") == [short, short]


@pytest.mark.parametrize("name", ["char", *BPE_NAMES])
def test_special_token_text_is_matched_literally_not_as_a_pattern(name: str) -> None:
    tokenizer = trained(name, ["a.c", "(x)"])
    ids = tokenizer.special_token_ids

    assert tokenizer.encode_string("a.c") == [ids["a.c"]]
    assert tokenizer.encode_string("(x)") == [ids["(x)"]]
    assert tokenizer.encode_string("abc") == trained(name, [], vocab_size=288).encode_string("abc")


# --- save and load

@pytest.mark.parametrize("name", ["char", *BPE_NAMES])
def test_a_loaded_tokenizer_keeps_its_special_tokens(name: str, tmp_path: Path) -> None:
    original = trained(name, [EOT, PAD])
    original.save(tmp_path / "tokenizer.json")

    loaded = load_tokenizer(tmp_path / "tokenizer.json")

    text = "the fox" + EOT + "the dog" + PAD
    assert loaded.special_tokens == [EOT, PAD]
    assert loaded.special_token_ids == original.special_token_ids
    assert loaded.get_vocab() == original.get_vocab()
    assert loaded.encode_string(text) == original.encode_string(text)
    assert loaded.decode_as_string(loaded.encode_string(text)) == text


@pytest.mark.parametrize("name", ["char", *BPE_NAMES])
def test_saving_a_loaded_tokenizer_with_special_tokens_gives_the_same_file(name: str, tmp_path: Path) -> None:
    trained(name, [EOT]).save(tmp_path / "a.json")
    load_tokenizer(tmp_path / "a.json").save(tmp_path / "b.json")
    assert (tmp_path / "a.json").read_text() == (tmp_path / "b.json").read_text()


def test_a_loaded_tokenizer_is_trained_so_it_rejects_new_special_tokens(tmp_path: Path) -> None:
    trained("char", [EOT]).save(tmp_path / "t.json")
    with pytest.raises(ValueError, match="before train"):
        load_tokenizer(tmp_path / "t.json").register_special_tokens([PAD])


def test_files_from_before_special_tokens_are_rejected(tmp_path: Path) -> None:
    trained("char", []).save(tmp_path / "t.json")
    saved = json.loads((tmp_path / "t.json").read_text())
    del saved["special_tokens"]
    saved["format"] = 1
    (tmp_path / "old.json").write_text(json.dumps(saved))

    with pytest.raises(ValueError, match="format"):
        load_tokenizer(tmp_path / "old.json")
