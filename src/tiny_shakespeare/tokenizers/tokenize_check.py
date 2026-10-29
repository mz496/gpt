from pathlib import Path

from src.tiny_shakespeare.tokenizers.byte_pair_encoding import BytePairEncoding
from src.tiny_shakespeare.utils import bytes_to_string

INPUT_PATH = Path(__file__).parent.parent / "input_data" / "tiny_shakespeare.txt"


def main():
    bpe = BytePairEncoding(300) # Temp: One merge pass.
    text = INPUT_PATH.read_text()
    bpe.train(text)
    tokens = bpe.encode_string(text)
    print(f"{len(text)} chars -> {len(tokens)} tokens")

    vocab = bpe.get_vocab()
    print("merges (unrolled to bytes):")
    for merge in bpe.merges:
        byte_values = list(vocab[merge.to_token])
        print(f"  {merge.to_token}: {merge.from_tokens[0]},{merge.from_tokens[1]} ({bytes_to_string(byte_values)!r})")

if __name__ == "__main__":
    main()
