
import argparse

from src.tiny_shakespeare.data.loader import get_sft_batch
from src.tiny_shakespeare.data.prepare import prepare_sft
from src.tiny_shakespeare.data.sft.reverse_source import ReverseSource
from src.tiny_shakespeare.data.sft.sft_data_source import SFTDataSource
from src.tiny_shakespeare.data.special_tokens import IM_END, IM_START
from src.tiny_shakespeare.models.gpt.model import GPT, load_checkpoint, load_checkpoint_tokenizer
from src.tiny_shakespeare.tokenizers.tokenizer import Tokenizer
from src.tiny_shakespeare.utils import get_device

SOURCES: dict[str, type[SFTDataSource]] = {
    "reverse": ReverseSource,
}

def check_checkpoint(model: GPT, tokenizer: Tokenizer) -> None:
    missing = [token for token in (IM_START, IM_END) if token not in tokenizer.special_tokens]
    if missing:
        raise ValueError(
            f"the checkpoint's tokenizer is missing the special tokens {missing} (it has {tokenizer.special_tokens}). "
            "Pretrain with a source that declares them, e.g. --source tiny-stories, and finetune that run."
        )
    if len(tokenizer.get_vocab()) != model.config.vocab_size:
        raise ValueError(f"the tokenizer's vocab size {len(tokenizer.get_vocab())} does not match the model's {model.config.vocab_size}")

def sft(checkpoint_run_id: str, data_source: SFTDataSource):
    device = get_device()
    print(f"available device: {device}")
    checkpoint = load_checkpoint(checkpoint_run_id, device)
    tokenizer = load_checkpoint_tokenizer(checkpoint_run_id)
    check_checkpoint(checkpoint, tokenizer)

    # Prepare examples: tokenize, mask, etc. (prepare_sft)
    # The checkpoint's context length is fixed (its position table has that many rows), so it decides how long an example can be.
    prepared_data = prepare_sft(data_source, tokenizer, block_size=checkpoint.config.block_size)
    print(prepared_data["train"])
    
    # TODO: complete:

    #get_sft_batch()

    #train()
    

def main():
    parser = argparse.ArgumentParser(description="Finetune a trained run's checkpoint.")
    parser.add_argument("run_id", help="run id of the checkpoint to finetune, e.g. 2026-10-28T11:34:42_abc123")
    parser.add_argument("--source", choices=list(SOURCES), default="reverse", help="which SFT dataset to finetune on")
    args = parser.parse_args()

    sft(args.run_id, SOURCES[args.source]())

if __name__ == "__main__":
    main()
