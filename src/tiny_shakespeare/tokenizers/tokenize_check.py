import argparse
import math
import time
from collections import Counter

from src.tiny_shakespeare.data.tiny_stories_source import TinyStoriesSource
from src.tiny_shakespeare.tokenizers.bpe_tokenizer import BPETokenizer
from src.tiny_shakespeare.tokenizers.regex_bpe_tokenizer import RegexBPETokenizer
from src.tiny_shakespeare.tokenizers.utils import TOKENIZER_NAMES, make_tokenizer
from src.tiny_shakespeare.data.tiny_shakespeare_source import TinyShakespeareSource
from src.tiny_shakespeare.utils import bits_per_char, bytes_to_string


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

    source = TinyStoriesSource()
    tokenizer = make_tokenizer(args.tokenizer, args.bpe_vocab_size)
    tokenizer.register_special_tokens(source.special_tokens)
    # Same train split (and train-only fitting) as models/gpt/train.py.
    # Each step says when it starts, since the first one waits on the download (the first time) and the rest can take a long time.
    print(f"reading the tokenizer text from {source.name} (downloads the dataset the first time)...", flush=True)
    start = time.perf_counter()
    train_data = source.get_tokenizer_text()
    print(f"read {len(train_data):,} chars in {time.perf_counter() - start:.1f}s", flush=True)

    print(f"training {args.tokenizer}...", flush=True)
    start = time.perf_counter()
    tokenizer.train(train_data)
    print(f"trained in {time.perf_counter() - start:.1f}s", flush=True)

    print("encoding the same text with it...", flush=True)
    start = time.perf_counter()
    tokens = tokenizer.encode_string(train_data)
    print(f"encoded in {time.perf_counter() - start:.1f}s", flush=True)
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
