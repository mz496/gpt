from typing import Iterator

import numpy as np
import pytest

from src.tiny_shakespeare.data.data_source import Split
from src.tiny_shakespeare.data.loader import IGNORE_INDEX, SftData
from src.tiny_shakespeare.data.prepare import PAD_ID, prepare_sft, prepare_sft_validation_slices
from src.tiny_shakespeare.data.sft.reverse_source import ReverseSource
from src.tiny_shakespeare.data.sft.sft_data_source import SFTDataSource, SftExample
from src.tiny_shakespeare.data.special_tokens import END_OF_TEXT, IM_END, IM_START
from src.tiny_shakespeare.tokenizers.char_tokenizer import CharTokenizer

BLOCK_SIZE = 64
UINT16 = np.dtype(np.uint16)


@pytest.fixture
def tokenizer() -> CharTokenizer:
    return CharTokenizer.from_state({"vocab": list("\nabcdefghijklmnopqrstuvwxyz"), "special_tokens": [END_OF_TEXT, IM_START, IM_END]})


class FakeSource(SFTDataSource):
    def __init__(self, train: list[SftExample], validation: list[SftExample]):
        self.splits = {"train": train, "validation": validation}

    def iter_examples(self, split: Split) -> Iterator[SftExample]:
        yield from self.splits[split]


def encode(tokenizer: CharTokenizer, examples: list[SftExample], block_size: int = BLOCK_SIZE) -> SftData:
    """The examples encoded as one slice, which is the public way to encode a list of examples."""
    return prepare_sft_validation_slices({"slice": examples}, tokenizer, block_size)["slice"]


def trained_text(tokenizer: CharTokenizer, row_targets: np.ndarray) -> str:
    """What a row's targets teach: the tokens that aren't ignored, as text."""
    return tokenizer.decode_as_string([int(t) for t in row_targets if t != IGNORE_INDEX])


# --- one example, and a list of them: cut, mask, pad

def test_each_example_is_cut_masked_and_padded_to_the_longest(tokenizer: CharTokenizer) -> None:
    short, long = SftExample(input="ab", output="ba"), SftExample(input="abc", output="cba")
    data = encode(tokenizer, [short, long])

    assert data.inputs.dtype == UINT16 and data.targets.dtype == np.int64
    full_short = tokenizer.encode_string(short.to_request_string()) + tokenizer.encode_string(short.to_response_string())
    full_long = tokenizer.encode_string(long.to_request_string()) + tokenizer.encode_string(long.to_response_string())
    width = len(full_long) - 1
    assert data.inputs.shape == data.targets.shape == (2, width)

    for row, (example, full) in enumerate([(short, full_short), (long, full_long)]):
        n_request = len(tokenizer.encode_string(example.to_request_string()))
        n_inputs = len(full) - 1
        # The inputs are the sequence without its last token, then padding.
        assert data.inputs[row, :n_inputs].tolist() == full[:-1]
        assert (data.inputs[row, n_inputs:] == PAD_ID).all()
        # The targets are the sequence without its first token, ignored over the request and over the padding.
        assert (data.targets[row, :n_request - 1] == IGNORE_INDEX).all()
        assert data.targets[row, n_request - 1:n_inputs].tolist() == full[n_request:]
        assert (data.targets[row, n_inputs:] == IGNORE_INDEX).all()
        # So what's learned is the response, and only the response.
        assert trained_text(tokenizer, data.targets[row]) == example.to_response_string()


def test_the_first_trained_position_is_the_last_token_of_the_request(tokenizer: CharTokenizer) -> None:
    data = encode(tokenizer, [SftExample(input="ab", output="ba")])
    first_trained = int(np.nonzero(data.targets[0] != IGNORE_INDEX)[0][0])
    # The model has read the whole request there, and its target is the first token of the answer.
    assert tokenizer.decode_as_string([int(data.inputs[0, first_trained])]) == "t"  # the last letter of "assistant"
    assert tokenizer.decode_as_string([int(data.targets[0, first_trained])]) == "b"


def test_an_example_that_does_not_fit_or_a_slice_with_no_examples_raises_naming_the_slice(tokenizer: CharTokenizer) -> None:
    with pytest.raises(ValueError, match=r"slice palindrome example 'aaaaaaaaaa' .*block_size 10"):
        prepare_sft_validation_slices({"palindrome": [SftExample(input="a" * 10, output="a" * 10)]}, tokenizer, 10)
    with pytest.raises(ValueError, match="no slice palindrome examples"):
        prepare_sft_validation_slices({"palindrome": []}, tokenizer, BLOCK_SIZE)


# --- the splits and the slices go through the same code

def test_prepare_sft_encodes_each_split_the_way_a_slice_is_encoded(tokenizer: CharTokenizer) -> None:
    train, validation = [SftExample("ab", "ba"), SftExample("abc", "cba")], [SftExample("xy", "yx")]
    data = prepare_sft(FakeSource(train, validation), tokenizer, BLOCK_SIZE)
    assert list(data) == ["train", "validation"]
    expected = prepare_sft_validation_slices({"train": train, "validation": validation}, tokenizer, BLOCK_SIZE)
    for split in ("train", "validation"):
        assert np.array_equal(data[split].inputs, expected[split].inputs) and np.array_equal(data[split].targets, expected[split].targets)


def test_an_example_is_encoded_the_same_in_training_and_in_a_slice(tokenizer: CharTokenizer) -> None:
    examples = [SftExample("ab", "ba"), SftExample("abc", "cba"), SftExample("q", "q")]
    trained_on = prepare_sft(FakeSource(examples, examples), tokenizer, BLOCK_SIZE)["train"]
    sliced = encode(tokenizer, examples)
    # If these differed, a score would measure something other than what was trained.
    assert np.array_equal(trained_on.inputs, sliced.inputs) and np.array_equal(trained_on.targets, sliced.targets)


def test_prepare_sft_validation_slices_gives_each_slice_its_own_rows_in_order_and_its_own_width(tokenizer: CharTokenizer) -> None:
    slices = {
        "short": [SftExample("a", "a"), SftExample("ab", "ba")],
        "long": [SftExample("abcdef", "fedcba"), SftExample("xy", "yx"), SftExample("pqr", "rqp")],
    }
    data = prepare_sft_validation_slices(slices, tokenizer, BLOCK_SIZE)

    assert list(data) == ["short", "long"]
    assert data["short"].inputs.shape[0] == 2 and data["long"].inputs.shape[0] == 3
    assert data["short"].inputs.shape[1] < data["long"].inputs.shape[1]  # a slice is as wide as its own longest example
    for name, examples in slices.items():
        assert data[name].inputs.dtype == UINT16 and data[name].targets.dtype == np.int64
        for row, example in enumerate(examples):
            assert trained_text(tokenizer, data[name].targets[row]) == example.to_response_string()


def test_the_pooled_validation_rows_are_the_rows_of_the_slices(tokenizer: CharTokenizer) -> None:
    # The loss is measured on the pooled validation split and the scores on the slices, so they must see the same rows.
    source = ReverseSource()
    pooled = prepare_sft(source, tokenizer, block_size=256)["validation"]
    slices = prepare_sft_validation_slices(source.get_validation_slices(), tokenizer, block_size=256)

    assert pooled.inputs.shape[0] == sum(data.inputs.shape[0] for data in slices.values())
    start = 0
    for data in slices.values():
        rows, width = data.inputs.shape
        # Each slice's rows, then (if the pooled array is wider) nothing but padding.
        assert np.array_equal(pooled.inputs[start:start + rows, :width], data.inputs)
        assert np.array_equal(pooled.targets[start:start + rows, :width], data.targets)
        assert (pooled.inputs[start:start + rows, width:] == PAD_ID).all()
        assert (pooled.targets[start:start + rows, width:] == IGNORE_INDEX).all()
        start += rows
