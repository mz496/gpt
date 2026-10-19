from src.tiny_shakespeare.main import decode, encode


def test_decode_encode_roundtrip() -> None:
    text = "First Citizen:\nBefore we proceed any further, hear me speak."
    vocab = sorted(set(text))

    assert decode(encode("ac", vocab), vocab) == "ac"
