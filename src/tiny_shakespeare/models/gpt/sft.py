
from src.tiny_shakespeare.data.loader import get_sft_batch
from src.tiny_shakespeare.data.prepare import prepare_sft


def sft():
    # TODO[sft]

    data = prepare_sft()
    
    get_sft_batch()

    # Pass batch to train

    

