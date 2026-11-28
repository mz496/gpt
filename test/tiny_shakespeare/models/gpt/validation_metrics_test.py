import pytest
import torch
import torch.nn.functional as F

from src.tiny_shakespeare.data.data_source import Split
from src.tiny_shakespeare.data.loader import IGNORE_INDEX, SftData
from src.tiny_shakespeare.data.prepare import prepare_sft_validation_slices
from src.tiny_shakespeare.data.sft.sft_data_source import SftExample
from src.tiny_shakespeare.data.special_tokens import END_OF_TEXT, IM_END, IM_START
from src.tiny_shakespeare.models.gpt.generate import generate_response
from src.tiny_shakespeare.models.gpt.model import GPT, GPTConfig
from src.tiny_shakespeare.models.gpt.train import run_metrics_hook
from src.tiny_shakespeare.models.gpt.validation_metrics import (
    RowScores, format_report, length_bin, make_validation_metrics_hook, pool, score_rows, score_validation, summarize, summarize_by_length_bin, validation_report,
)
from src.tiny_shakespeare.tokenizers.char_tokenizer import CharTokenizer

II = IGNORE_INDEX
VOCAB = 5


def one_hot_logits(predictions: list[list[int]], vocab: int = VOCAB, strength: float = 8.0) -> torch.Tensor:
    """Logits whose top choice at every position is the given token, by a margin of `strength`."""
    return F.one_hot(torch.tensor(predictions), vocab).float() * strength


# --- score_rows: the pure comparison

def test_a_row_passes_if_every_response_position_is_right_and_only_those_count() -> None:
    targets = torch.tensor([
        [II, II, 1, 2, 3],   # all right
        [II, 4, 1, 2, II],   # wrong at its 2nd response token
        [II, 0, 1, II, II],  # wrong only where the target is ignored
        [II, II, 2, 2, 2],   # wrong at its first response token
    ])
    predictions = [
        [0, 0, 1, 2, 3],
        [0, 4, 0, 2, 0],
        [3, 0, 1, 3, 3],
        [0, 0, 1, 2, 2],
    ]
    scores = score_rows(one_hot_logits(predictions), targets)

    assert scores.passed.tolist() == [True, False, True, False]  # pyright: ignore[reportUnknownMemberType] Bad torch type stubs.
    assert scores.n_tokens.tolist() == [3, 3, 2, 3]  # pyright: ignore[reportUnknownMemberType] Bad torch type stubs.
    assert scores.n_right.tolist() == [3, 2, 2, 2]  # pyright: ignore[reportUnknownMemberType] Bad torch type stubs.
    # Counted within the response: 0 is its first token.
    assert scores.first_error.tolist() == [-1, 1, -1, 0]  # pyright: ignore[reportUnknownMemberType] Bad torch type stubs.


def test_the_loss_is_the_cross_entropy_summed_over_the_response_positions_only() -> None:
    torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType] Bad torch type stubs.
    logits = torch.randn(3, 6, VOCAB)
    targets = torch.tensor([[II, 1, 2, 3, II, II], [II, II, II, 4, 0, 1], [0, 1, 2, 3, 4, 0]])
    scores = score_rows(logits, targets)
    for row in range(3):
        kept = targets[row] != II
        expected = F.cross_entropy(logits[row][kept], targets[row][kept], reduction="sum")
        assert scores.loss_sum[row].item() == pytest.approx(expected.item(), rel=1e-5)


def test_a_row_with_nothing_to_score_raises() -> None:
    with pytest.raises(ValueError, match="no response positions"):
        score_rows(one_hot_logits([[0, 1, 2]]), torch.tensor([[II, II, II]]))


def test_low_precision_logits_are_scored_in_float32() -> None:
    targets = torch.tensor([[II, 1, 2]])
    scores = score_rows(one_hot_logits([[0, 1, 2]]).bfloat16(), targets)
    assert scores.loss_sum.dtype == torch.float32 and bool(scores.passed[0])


# --- summaries

def scores_of(passed: list[bool], n_tokens: list[int], n_right: list[int], loss_sum: list[float]) -> RowScores:
    return RowScores(
        passed=torch.tensor(passed), n_tokens=torch.tensor(n_tokens), n_right=torch.tensor(n_right),
        first_error=torch.full((len(passed),), -1), loss_sum=torch.tensor(loss_sum),
    )


def test_summarize_counts_rows_for_exact_match_and_tokens_for_accuracy_and_loss() -> None:
    # A short row that passes and a long one that doesn't: the long one weighs more in the token averages, and not more in exact_match.
    summary = summarize(scores_of(passed=[True, False], n_tokens=[2, 8], n_right=[2, 4], loss_sum=[0.2, 7.8]))
    assert summary.count == 2
    assert summary.exact_match == pytest.approx(0.5)
    assert summary.token_acc == pytest.approx(6 / 10)
    assert summary.loss == pytest.approx(8.0 / 10)


def test_summarize_raises_on_no_rows() -> None:
    with pytest.raises(ValueError, match="no rows"):
        summarize(scores_of([], [], [], []))


def test_pool_puts_the_rows_one_after_the_other() -> None:
    first, second = scores_of([True], [2], [2], [0.1]), scores_of([False, True], [3, 4], [1, 4], [2.0, 0.3])
    pooled = pool([first, second])
    assert pooled.passed.tolist() == [True, False, True] and pooled.n_tokens.tolist() == [2, 3, 4]  # pyright: ignore[reportUnknownMemberType] Bad torch type stubs.
    with pytest.raises(ValueError):
        pool([])


@pytest.mark.parametrize("length, label", [(1, "1-10"), (10, "1-10"), (11, "11-20"), (50, "41-50"), (51, "51-60")])
def test_length_bins(length: int, label: str) -> None:
    assert length_bin(length) == label


def test_length_bin_rejects_an_empty_string() -> None:
    with pytest.raises(ValueError):
        length_bin(0)


def test_summarize_by_length_bin_groups_rows_and_skips_empty_bins() -> None:
    scores = scores_of(passed=[True, False, True, True], n_tokens=[2, 2, 2, 2], n_right=[2, 1, 2, 2], loss_sum=[0.0, 3.0, 0.0, 0.0])
    # Lengths 3 and 10 share a bin, 25 is in its own, and nothing is between 10 and 25 (so there is no 11-20).
    by_bin = summarize_by_length_bin(scores, [3, 10, 25, 28])

    assert list(by_bin) == ["1-10", "21-30"]
    assert by_bin["1-10"].count == 2 and by_bin["1-10"].exact_match == pytest.approx(0.5)
    assert by_bin["21-30"].count == 2 and by_bin["21-30"].exact_match == pytest.approx(1.0)
    with pytest.raises(ValueError, match="lengths"):
        summarize_by_length_bin(scores, [3, 10])


# --- score_validation: a model over SftData

@pytest.fixture
def tokenizer() -> CharTokenizer:
    return CharTokenizer.from_state({"vocab": list("\nabcdefghijklmnopqrstuvwxyz"), "special_tokens": [END_OF_TEXT, IM_START, IM_END]})


def make_model(tokenizer: CharTokenizer) -> GPT:
    torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType] Bad torch type stubs.
    return GPT(GPTConfig(vocab_size=len(tokenizer.get_vocab()), n_embd=32, n_head=2, n_layer=2, block_size=64, dropout=0.0))


def encode(tokenizer: CharTokenizer, examples: list[SftExample]) -> SftData:
    return prepare_sft_validation_slices({"slice": examples}, tokenizer, block_size=64)["slice"]


def train_on_first_rows(model: GPT, data: SftData, n_rows: int, steps: int = 400) -> None:
    """Trains the model on the first rows of the data alone until it has memorized them."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    inputs, targets = torch.from_numpy(data.inputs[:n_rows].astype("int64")), torch.from_numpy(data.targets[:n_rows])  # pyright: ignore[reportUnknownMemberType] Bad torch type stubs.
    model.train()
    for _ in range(steps):
        optimizer.zero_grad()
        _, loss = model(inputs, targets)
        assert loss is not None
        loss.backward()
        optimizer.step()  # pyright: ignore[reportUnknownMemberType] Bad torch type stubs.


def test_chunking_does_not_change_the_scores_and_the_model_is_left_as_it_was(tokenizer: CharTokenizer) -> None:
    model = make_model(tokenizer)
    data = encode(tokenizer, [SftExample(s, s[::-1]) for s in ["abc", "cab", "bca", "abba", "cbbc", "aabb", "q"]])

    model.train()
    whole = score_validation(model, data, "cpu", chunk_size=100)
    assert model.training, "the model must be back in the mode it was in"
    model.eval()
    for chunk_size in (1, 3):
        chunked = score_validation(model, data, "cpu", chunk_size=chunk_size)
        assert not model.training
        assert torch.equal(chunked.passed, whole.passed) and torch.equal(chunked.first_error, whole.first_error)
        assert torch.allclose(chunked.loss_sum, whole.loss_sum, rtol=1e-4)
    assert not whole.loss_sum.requires_grad
    with pytest.raises(ValueError, match="chunk_size"):
        score_validation(model, data, "cpu", chunk_size=0)


def test_teacher_forced_pass_is_exactly_whether_greedy_generation_gives_the_right_answer(tokenizer: CharTokenizer) -> None:
    # The claim the whole module rests on. A model that has memorized some strings passes on those and fails on others, so this has both outcomes to agree on.
    memorized = [SftExample(s, s[::-1]) for s in ["abca", "bcab", "cabc", "abba", "caac", "bcbc"]]
    unseen = [SftExample(s, s[::-1]) for s in ["acbc", "babc", "ccab", "bacb", "abbc", "cbab"]]
    data = encode(tokenizer, memorized + unseen)
    model = make_model(tokenizer)

    train_on_first_rows(model, data, len(memorized))

    scores = score_validation(model, data, "cpu")
    generated_right = [
        generate_response(model, tokenizer, example.input, max_tokens=30, greedy=True, device="cpu") == example.to_response_string()
        for example in memorized + unseen
    ]
    assert scores.passed.tolist() == generated_right  # pyright: ignore[reportUnknownMemberType] Bad torch type stubs.
    assert any(generated_right) and not all(generated_right), "both outcomes should occur, or the agreement says little"


# --- the hook

MEMORIZED = [SftExample(s, s[::-1]) for s in ["abca", "bcab", "cabc", "abba", "caac", "bcbc"]]
UNSEEN = [SftExample(s, s[::-1]) for s in ["acbc", "babc", "ccab", "bacb", "abbc", "cbab"]]
LOSSES: dict[Split, float] = {"train": 1.0, "validation": 1.0}


@pytest.fixture
def memorizing_model_and_slices(tokenizer: CharTokenizer) -> tuple[GPT, dict[str, SftData]]:
    """A model that has memorized one slice and not the other, and the two slices."""
    slices = prepare_sft_validation_slices({"memorized": MEMORIZED, "unseen": UNSEEN}, tokenizer, block_size=64)
    model = make_model(tokenizer)
    train_on_first_rows(model, slices["memorized"], len(MEMORIZED))
    return model, slices


def test_the_hook_reports_every_slice_and_the_pool_apart_from_the_losses_train_logs(memorizing_model_and_slices: tuple[GPT, dict[str, SftData]]) -> None:
    model, slices = memorizing_model_and_slices
    metrics = make_validation_metrics_hook(slices, "cpu")(0, model, LOSSES)

    assert set(metrics) == {
        "validation/memorized/exact_match", "validation/memorized/token_acc", "validation/memorized/loss",
        "validation/unseen/exact_match", "validation/unseen/token_acc", "validation/unseen/loss",
        "validation/exact_match", "validation/token_acc",
    }
    assert not set(metrics) & {"train/loss", "validation/loss"}, "train() would raise on a key that it logs itself"
    assert all(type(value) is float for value in metrics.values())
    # The slice it has memorized is told apart from the one it hasn't, and the pool is the two together (the same number of rows each).
    assert metrics["validation/memorized/exact_match"] == 1.0 and metrics["validation/memorized/loss"] < metrics["validation/unseen/loss"]
    assert metrics["validation/unseen/exact_match"] < 1.0
    assert metrics["validation/exact_match"] == pytest.approx((metrics["validation/memorized/exact_match"] + metrics["validation/unseen/exact_match"]) / 2)


def test_the_hook_gives_the_same_numbers_through_run_metrics_hook_and_leaves_the_model_in_train_mode(memorizing_model_and_slices: tuple[GPT, dict[str, SftData]]) -> None:
    model, slices = memorizing_model_and_slices
    hook = make_validation_metrics_hook(slices, "cpu")
    model.eval()
    direct = hook(0, model, LOSSES)

    model.train()
    guarded = run_metrics_hook(hook, 0, model, LOSSES, None)  # the way train() calls it
    assert model.training
    assert guarded == direct


def test_without_slices_the_hook_scores_the_validation_split_alone(tokenizer: CharTokenizer) -> None:
    validation = prepare_sft_validation_slices({"validation": MEMORIZED + UNSEEN}, tokenizer, block_size=64)["validation"]
    model = make_model(tokenizer)
    train_on_first_rows(model, validation, len(MEMORIZED))

    metrics = make_validation_metrics_hook({}, "cpu", validation=validation)(0, model, LOSSES)

    assert set(metrics) == {"validation/exact_match", "validation/token_acc"}
    assert 0.0 < metrics["validation/exact_match"] < 1.0


def test_the_hook_needs_something_to_score() -> None:
    with pytest.raises(ValueError, match="nothing to score"):
        make_validation_metrics_hook({}, "cpu")


def test_the_hook_prints_one_line_with_each_slice_so_that_progress_shows_in_the_terminal(
    memorizing_model_and_slices: tuple[GPT, dict[str, SftData]], capsys: pytest.CaptureFixture[str],
) -> None:
    model, slices = memorizing_model_and_slices
    make_validation_metrics_hook(slices, "cpu")(50, model, LOSSES)
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("step=0050 exact_match: memorized 1.000 | unseen ") and lines[0].split("all ")[-1].replace(".", "").isdigit()


# --- the report

def test_the_report_gives_counts_by_slice_by_length_bin_and_overall() -> None:
    def scores(passed: list[bool]) -> RowScores:
        return scores_of(passed=passed, n_tokens=[2] * len(passed), n_right=[2] * len(passed), loss_sum=[0.0] * len(passed))

    report = format_report(
        {"palindrome": scores([True, False, True]), "random": scores([False, False])},
        # palindrome: two strings of lengths 3 and 10 (one bin), one of 25; random: one of 12 and one of 14.
        {"palindrome": [3, 10, 25], "random": [12, 14]},
    )
    assert report.splitlines() == [
        "validation, rows passed / rows:",
        "  palindrome  2/3",
        "    1-10        1/2",
        "    21-30       1/1",
        "  random      0/2",
        "    11-20       0/2",
        "  all         2/5",
    ]


def test_the_report_of_a_model_counts_the_rows_it_passes(memorizing_model_and_slices: tuple[GPT, dict[str, SftData]]) -> None:
    model, slices = memorizing_model_and_slices
    lengths = {"memorized": [len(e.input) for e in MEMORIZED], "unseen": [len(e.input) for e in UNSEEN]}
    lines = validation_report(model, slices, lengths, "cpu").splitlines()
    # The memorized slice passes whole (all four-letter strings, so one bin), and the pool is that and whatever of the unseen one passes.
    assert lines[1].split() == ["memorized", "6/6"] and lines[2].split() == ["1-10", "6/6"]
    assert lines[-1].startswith("  all") and lines[-1].endswith("/12")
