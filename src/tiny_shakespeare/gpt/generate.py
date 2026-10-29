import argparse

import torch

from ..tokenizers.utils import make_tokenizer
from ..utils import get_tiny_shakespeare_data
from .model import get_device, load_checkpoint

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a trained run's checkpoint")
    parser.add_argument("run_id", help="run id printed by training, e.g. 2026-10-28T11:34:42_abc123")
    parser.add_argument("--generate-seed", type=int, default=1337, help="seed for the sampling")
    parser.add_argument("--max-tokens", type=int, default=1000, help="number of tokens to generate")
    return parser.parse_args()

def main():
    args = parse_args()
    device = get_device()
    print(f"available device: {device}")

    model, tokenizer_name, bpe_vocab_size = load_checkpoint(args.run_id, device)
    # The tokenizer is retrained on the data, which is deterministic for both tokenizers.
    train_data, validation_data = get_tiny_shakespeare_data()
    tokenizer = make_tokenizer(tokenizer_name, bpe_vocab_size)
    tokenizer.train(train_data + validation_data)
    if len(tokenizer.get_vocab()) != model.config.vocab_size:
        raise ValueError(f"vocab size {len(tokenizer.get_vocab())} does not match the checkpoint's {model.config.vocab_size}")

    torch.manual_seed(args.generate_seed)  # pyright: ignore[reportUnknownMemberType] Bad torch type stubs.
    idx = torch.zeros((1, 1), dtype=torch.long).to(device)
    with torch.no_grad():
        result_tokens = model.generate(idx, max_tokens=args.max_tokens)
    result = tokenizer.decode_as_string(result_tokens[0].tolist())  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType] Bad torch type stubs.
    print(result)

if __name__ == "__main__":
    main()
