import argparse

import torch

from ..tokenizers.char_encoding import CharEncoding
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

    # The tokenizer is retrained on the data, which is deterministic for the char tokenizer.
    train_data, validation_data = get_tiny_shakespeare_data()
    tokenizer = CharEncoding()
    tokenizer.train(train_data + validation_data)
    model = load_checkpoint(args.run_id, device)
    if tokenizer.get_vocab_size() != model.config.vocab_size:
        raise ValueError(f"vocab size {tokenizer.get_vocab_size()} does not match the checkpoint's {model.config.vocab_size}")

    torch.manual_seed(args.generate_seed)  # pyright: ignore[reportUnknownMemberType] Bad torch type stubs.
    idx = torch.zeros((1, 1), dtype=torch.long).to(device)
    with torch.no_grad():
        result_tokens = model.generate(idx, max_tokens=args.max_tokens)
    result = tokenizer.decode_as_string(result_tokens[0].tolist())  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType] Bad torch type stubs.
    print(result)

if __name__ == "__main__":
    main()
