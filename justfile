# Separates output by tests and shows one test result per line
test *args:
    uv run --locked pytest -vrP {{args}}

# Trains and generates from the bigram language model
bigram *args:
    uv run --locked -m src.tiny_shakespeare.bigram {{args}}

# Trains the decoder-only transformer
gpt-train *args:
    uv run --locked -m src.tiny_shakespeare.gpt.train {{args}}

# Generates from a specified training run ID checkpoint
gpt-generate run_id *args:
    uv run --locked -m src.tiny_shakespeare.gpt.generate {{run_id}} {{args}}

# Runs diagnostics on the specified tokenizer and parameters
# (check merges for BPE-based; check compression ratios, etc)
tokenize-check *args:
    uv run --locked -m src.tiny_shakespeare.tokenizers.tokenize_check {{args}}

# Kicks off a few runs and plots on W&B
compare-tokenizers *args:
    uv run --locked -m scripts.compare_tokenizers {{args}}

# --- W&B helpers ---
wandb-create-view *args:
    @uv run --locked -m scripts.wandb_create_view {{args}}
wandb-add-run-to-view run_id view_id *args:
    uv run --locked -m scripts.wandb_add_run_to_view {{run_id}} {{view_id}} {{args}}
wandb-rename-run run_id name *args:
    uv run --locked -m scripts.wandb_rename_run {{run_id}} "{{name}}" {{args}}
wandb-set-color run_id view_id color *args:
    uv run --locked -m scripts.wandb_set_color {{run_id}} {{view_id}} "{{color}}" {{args}}
