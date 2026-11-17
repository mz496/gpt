import math
from typing import Any

import numpy as np
import torch

from ...tokenizers.char_tokenizer import CharTokenizer
from ...tokenizers.tokenizer import Tokenizer
from ...data.loader import get_batch, get_tokenizer_text
from ...data.prepare import PreparedData, prepare
from ...utils import get_device
from .model import BigramModel

# --- hyperparameters ---
# Training
# how many independent sequences will we look at in parallel?
BATCH_SIZE = 4
# what is the max context length for predictions?
BLOCK_SIZE = 8
LEARNING_RATE = 1e-3
# for loss estimation
EVAL_ITERS = 200

@torch.no_grad
def estimate_loss(model: torch.nn.Module, tokenized_train_data: np.ndarray[Any, Any], tokenized_val_data: np.ndarray[Any, Any], device: str) -> dict[str, float]:
    """
    Estimate loss over train, validation based on EVAL_ITERS batches.
    """
    model.eval()
    result: dict[str, float] = {}
    for dataset_type, data in [("train", tokenized_train_data), ("validation", tokenized_val_data)]:
        # Compute the mean per dataset type.
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            contexts, targets = get_batch(data, BATCH_SIZE, BLOCK_SIZE)
            contexts, targets = contexts.to(device), targets.to(device)
            _logits, loss = model(contexts, targets)
            losses[k] = loss
        result[dataset_type] = losses.mean().item()
    model.train()
    return result


def train(step_count: int, data: PreparedData, tokenizer: Tokenizer, device: str) -> BigramModel:
    model = BigramModel(len(tokenizer.get_vocab())).to(device)
    # A reasonable lr for adam is 1e-3 to 1e-4.
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    for step in range(step_count):
        # Zero out gradients for this iteration.
        model.zero_grad()

        contexts, targets = get_batch(data.train, batch_size=BATCH_SIZE, block_size=BLOCK_SIZE)
        contexts, targets = contexts.to(device), targets.to(device)
        _logits, loss = model(contexts, targets)
        # expect logits shape [B, T, C]
        # An untrained model is near-uniform over C tokens, so loss should start near ~ln(C)
        if step % 100 == 0:
            print(f"step={step:04d} loss={estimate_loss(model, data.train, data.validation, device)} expected init ~{math.log(len(tokenizer.get_vocab())):.3f}")

        # Write in gradients based on how bad this iteration was.
        loss.backward()

        # Update weights with those gradients.
        optimizer.step() # pyright: ignore[reportUnknownMemberType] Bad torch type stubs.
    return model

def main():
    torch.manual_seed(1337)  # pyright: ignore[reportUnknownMemberType] Bad torch type stubs.
    tokenizer = CharTokenizer()
    tokenizer.train(get_tokenizer_text("tiny-shakespeare"))
    data = prepare("tiny-shakespeare", tokenizer)
    device = get_device()
    print(f"available device: {device}")

    model = train(10000, data, tokenizer, device)

    idx = torch.zeros((1, 1), dtype=torch.long).to(device)
    result_tokens = model.generate(idx, max_tokens=1000)
    result = tokenizer.decode_as_string(result_tokens[0].tolist())  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType] Bad torch type stubs.
    print(result)

if __name__ == "__main__":
    main()