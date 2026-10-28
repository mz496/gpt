from src.tiny_shakespeare.tokenizers.char_encoding import CharEncoding


def test_decode_encode_roundtrip() -> None:
    text = "First Citizen:\nBefore we proceed any further, hear me speak."
    tokenizer = CharEncoding()
    tokenizer.train(text)

    assert tokenizer.decode_as_string(tokenizer.encode_string("ac")) == "ac"
