test *args:
    uv run --locked pytest {{args}}

bigram *args:
    uv run --locked -m src.tiny_shakespeare.bigram {{args}}

gpt-train *args:
    uv run --locked -m src.tiny_shakespeare.gpt.train {{args}}

gpt-generate run_id *args:
    uv run --locked -m src.tiny_shakespeare.gpt.generate {{run_id}} {{args}}
