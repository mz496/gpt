import argparse

import torch

from ...data.sft.sft_data_source import SftExample
from ...tokenizers.tokenizer import Tokenizer
from ...utils import get_device
from .model import GPT, load_checkpoint, load_checkpoint_tokenizer

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a trained run's checkpoint")
    parser.add_argument("run_id", help="run id printed by training, e.g. 2026-10-28T11:34:42_abc123")
    parser.add_argument("--generate-seed", type=int, default=1337, help="seed for the sampling")
    parser.add_argument("--max-tokens", type=int, default=1000, help="number of tokens to generate")
    parser.add_argument("--prompt", default=None, help="for a finetuned checkpoint: the input to answer, e.g. a string to reverse. It's wrapped in the request template, and generation stops at the end of the response")
    parser.add_argument("--greedy", action="store_true", help="take the most likely token each step instead of sampling, so the same prompt always gives the same output")
    return parser.parse_args()

def generate_response(model: GPT, tokenizer: Tokenizer, prompt: str, max_tokens: int, greedy: bool, device: str) -> str:
    """
    The model's answer to `prompt`, built with the SftExample template.
    Generation stops at the end of the response (or at a special token).
    Returns what the model generated after the request, including that last token.
    """
    request_tokens = tokenizer.encode_string(SftExample(input=prompt, output="").to_request_string())
    # The model only sees its last block_size tokens, so a longer request would silently lose its start.
    if len(request_tokens) >= model.config.block_size:
        raise ValueError(f"the request is {len(request_tokens)} tokens, which leaves no room to answer in the model's block_size {model.config.block_size}")
    terminator = tokenizer.encode_string(SftExample(input="", output="").to_response_string())
    if len(terminator) != 1:
        raise ValueError(f"the end of a response is {len(terminator)} tokens, but only a single token can be a stop token")

    idx = torch.tensor([request_tokens], dtype=torch.long, device=device)
    with torch.no_grad():
        result_tokens = model.generate(idx, max_tokens=max_tokens, stop_token_ids=set(tokenizer.special_token_ids.values()) | set(terminator), greedy=greedy)
    return tokenizer.decode_as_string(result_tokens[0, len(request_tokens):].tolist())  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType] Bad torch type stubs.

def main():
    args = parse_args()
    device = get_device()
    print(f"available device: {device}")

    model = load_checkpoint(args.run_id, device, flag_use_pytorch_attention_impl=True)
    # Raises if the checkpoint has no tokenizer.
    tokenizer = load_checkpoint_tokenizer(args.run_id)
    if len(tokenizer.get_vocab()) != model.config.vocab_size:
        raise ValueError(f"vocab size {len(tokenizer.get_vocab())} does not match the checkpoint's {model.config.vocab_size}")

    torch.manual_seed(args.generate_seed)  # pyright: ignore[reportUnknownMemberType] Bad torch type stubs.
    if args.prompt is not None:
        response = generate_response(model, tokenizer, args.prompt, args.max_tokens, args.greedy, device)
        print(f"prompt:   {args.prompt}")
        print(f"response: {response.rstrip(chr(10))}")
        return
    idx = torch.zeros((1, 1), dtype=torch.long).to(device)
    with torch.no_grad():
        # Stop once the model emits a special token (e.g. end-of-text), if the tokenizer has any.
        result_tokens = model.generate(idx, max_tokens=args.max_tokens, stop_token_ids=set(tokenizer.special_token_ids.values()), greedy=args.greedy)
    result = tokenizer.decode_as_string(result_tokens[0].tolist())  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType] Bad torch type stubs.
    print(result)

if __name__ == "__main__":
    main()
