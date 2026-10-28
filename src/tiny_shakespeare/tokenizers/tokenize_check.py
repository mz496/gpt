from pathlib import Path

from src.tiny_shakespeare.tokenizers.byte_pair_encoding import BytePairEncoding

INPUT_PATH = Path(__file__).parent.parent / "input_data" / "tiny_shakespeare.txt"


def main():
    bpe = BytePairEncoding(257) # Temp: One merge pass.
    text = INPUT_PATH.read_text()
    bpe.train(text)
    tokens = bpe.encode_string(text)
    print(f"{len(text)} chars -> {len(tokens)} tokens")

if __name__ == "__main__":
    main()
