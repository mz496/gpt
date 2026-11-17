from src.tiny_shakespeare.data.loader import take_fraction


def test_take_fraction_one_returns_the_text_unchanged() -> None:
    text = "".join(chr(97 + i % 26) for i in range(5_321))
    assert take_fraction(text, 1.0) == text


def chunks_of(s: str) -> set[str]:
    return {s[i:i + 1000] for i in range(0, len(s), 1000)}


def test_take_fraction_is_nested_and_deterministic() -> None:
    text = "".join(f"{i:04d}" for i in range(2_500))  # 10,000 chars, each 1000-char chunk distinguishable
    eighth, quarter, half = (take_fraction(text, f) for f in (0.125, 0.25, 0.5))
    assert len(eighth) < len(quarter) < len(half) < len(text)
    assert chunks_of(eighth) <= chunks_of(quarter) <= chunks_of(half) <= chunks_of(text)
    assert take_fraction(text, 0.25) == quarter


def test_take_fraction_keeps_chunks_in_original_order() -> None:
    text = "".join(f"{i:04d}" for i in range(2_500))
    sample = take_fraction(text, 0.5)
    starts = [int(sample[i:i + 4]) for i in range(0, len(sample), 1000)]
    assert starts == sorted(starts)


def test_take_fraction_is_not_a_prefix() -> None:
    text = "".join(f"{i:04d}" for i in range(2_500))
    assert not text.startswith(take_fraction(text, 0.5))
