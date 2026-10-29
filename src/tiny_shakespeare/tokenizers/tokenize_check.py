import argparse
import math
from collections import Counter

from src.tiny_shakespeare.tokenizers.bpe_tokenizer import BPETokenizer
from src.tiny_shakespeare.tokenizers.regex_bpe_tokenizer import RegexBPETokenizer
from src.tiny_shakespeare.tokenizers.utils import TOKENIZER_NAMES, make_tokenizer
from src.tiny_shakespeare.utils import bits_per_char, bytes_to_string, get_tiny_shakespeare_data


def unigram_entropy(tokens: list[int]) -> float:
    """Entropy in nats of the token frequencies, ignoring context: -sum p ln p.
    i.e. If we just predicted the next char solely based on frequencies.
    """
    total = len(tokens)
    return -sum(c / total * math.log(c / total) for c in Counter(tokens).values())


def main():
    parser = argparse.ArgumentParser(description="Train a tokenizer on tiny shakespeare and print stats about it.")
    parser.add_argument("--tokenizer", choices=TOKENIZER_NAMES, default="bpe", help="which tokenizer to check")
    parser.add_argument("--bpe-vocab-size", type=int, default=512, help="target vocab size when --tokenizer=bpe or regex_bpe")
    parser.add_argument("--bpe-show-merges", action="store_true", help="print the learned merges when --tokenizer is a BPE type")
    args = parser.parse_args()

    tokenizer = make_tokenizer(args.tokenizer, args.bpe_vocab_size)
    # Same train split (and train-only fitting) as gpt/train.py.
    train_data, _validation_data = get_tiny_shakespeare_data()
    tokenizer.train(train_data)
    tokens = tokenizer.encode_string(train_data)
    print(f"compression ratio: {len(train_data)} chars -> {len(tokens)} tokens ({len(train_data) / len(tokens):.3f} chars/token)")

    vocab_size = len(tokenizer.get_vocab())
    print(f"vocab size V={vocab_size}")

    # Model has learned nothing; every char is equally likely; this is the starting loss
    uniform = math.log(vocab_size)
    print(f"uniform loss ln(V) = {uniform:.3f} nats/token, {bits_per_char(uniform, len(tokens), len(train_data)):.3f} bits/char")

    # Model has learned token frequencies without context; this is the plateau loss
    entropy = unigram_entropy(tokens)
    print(f"unigram entropy =    {entropy:.3f} nats/token, {bits_per_char(entropy, len(tokens), len(train_data)):.3f} bits/char")

    if args.bpe_show_merges and isinstance(tokenizer, (BPETokenizer, RegexBPETokenizer)):
        vocab = tokenizer.get_vocab()
        print("merges (unrolled to bytes):")
        for merge in tokenizer.merges:
            byte_values = list(vocab[merge.to_token])
            print(f"  {merge.to_token}: {merge.from_tokens[0]},{merge.from_tokens[1]} ({bytes_to_string(byte_values)!r})")

if __name__ == "__main__":
    main()
