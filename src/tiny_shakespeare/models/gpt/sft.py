
import argparse

from src.tiny_shakespeare.config import CONFIG_NAMES, TrainConfig, load_config
from src.tiny_shakespeare.data.loader import get_sft_batch
from src.tiny_shakespeare.data.prepare import prepare_sft
from src.tiny_shakespeare.data.sft.reverse_source import ReverseSource
from src.tiny_shakespeare.data.sft.sft_data_source import SFTDataSource
from src.tiny_shakespeare.models.gpt.model import load_checkpoint, load_checkpoint_tokenizer
from src.tiny_shakespeare.utils import get_device

SOURCES: dict[str, type[SFTDataSource]] = {
    "reverse": ReverseSource,
}

def sft(checkpoint_run_id: str, cfg: TrainConfig, data_source: SFTDataSource):
    device = get_device()
    print(f"available device: {device}")
    checkpoint = load_checkpoint(checkpoint_run_id, device)
    tokenizer = load_checkpoint_tokenizer(checkpoint_run_id)

    # Prepare examples: tokenize, mask, etc. (prepare_sft)
    prepared_data = prepare_sft(data_source, tokenizer, block_size=cfg.block_size)
    print(prepared_data.train)
    
    # TODO: complete:
    
    #get_sft_batch()

    #train()
    

def main():
    parser = argparse.ArgumentParser(description="Finetune a trained run's checkpoint.")
    parser.add_argument("run_id", help="run id of the checkpoint to finetune, e.g. 2026-10-28T11:34:42_abc123")
    parser.add_argument("--config", choices=CONFIG_NAMES, default="default", help="which config in configs/ to use")
    parser.add_argument("--source", choices=list(SOURCES), default="reverse", help="which SFT dataset to finetune on")
    args = parser.parse_args()
    cfg = load_config(args.config)

    sft(args.run_id, cfg, SOURCES[args.source]())

if __name__ == "__main__":
    main()
