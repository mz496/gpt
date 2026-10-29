import argparse
import math
from collections import Counter
from pathlib import Path

from src.tiny_shakespeare.tokenizers.bpe_tokenizer import BPETokenizer
from src.tiny_shakespeare.tokenizers.utils import TOKENIZER_NAMES, make_tokenizer
from src.tiny_shakespeare.utils import bits_per_char, bytes_to_string

INPUT_PATH = Path(__file__).parent.parent / "input_data" / "tiny_shakespeare.txt"


def unigram_entropy(tokens: list[int]) -> float:
    """Entropy in nats of the token frequencies, ignoring context: -sum p ln p.
    i.e. If we just predicted the next char solely based on frequencies.
    """
    total = len(tokens)
    return -sum(c / total * math.log(c / total) for c in Counter(tokens).values())


def main():
    parser = argparse.ArgumentParser(description="Train a tokenizer on tiny shakespeare and print stats about it.")
    parser.add_argument("--tokenizer", choices=TOKENIZER_NAMES, default="bpe", help="which tokenizer to check")
    parser.add_argument("--bpe-vocab-size", type=int, default=512, help="target vocab size when --tokenizer=bpe")
    args = parser.parse_args()

    tokenizer = make_tokenizer(args.tokenizer, args.bpe_vocab_size)
    text = INPUT_PATH.read_text()
    tokenizer.train(text)
    tokens = tokenizer.encode_string(text)
    print(f"{len(text)} chars -> {len(tokens)} tokens ({len(text) / len(tokens):.3f} chars/token)")

    # A model that ignores context and just predicts token frequencies plateaus at this loss.
    vocab_size = len(tokenizer.get_vocab())
    uniform = math.log(vocab_size)
    print(f"vocab size {vocab_size}: uniform loss ln(V)={uniform:.3f} nats/token, {bits_per_char(uniform, len(tokens), len(text)):.3f} bits/char")
    entropy = unigram_entropy(tokens)
    print(f"unigram entropy: {entropy:.3f} nats/token, {entropy * len(tokens) / len(text):.3f} nats/char, {bits_per_char(entropy, len(tokens), len(text)):.3f} bits/char")

    if isinstance(tokenizer, BPETokenizer):
        vocab = tokenizer.get_vocab()
        print("merges (unrolled to bytes):")
        for merge in tokenizer.merges:
            byte_values = list(vocab[merge.to_token])
            print(f"  {merge.to_token}: {merge.from_tokens[0]},{merge.from_tokens[1]} ({bytes_to_string(byte_values)!r})")

if __name__ == "__main__":
    main()
