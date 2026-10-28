# Separates output by tests and shows one test result per line
test *args:
    uv run --locked pytest -vrP {{args}}

bigram *args:
    uv run --locked -m src.tiny_shakespeare.bigram {{args}}

gpt-train *args:
    uv run --locked -m src.tiny_shakespeare.gpt.train {{args}}

gpt-generate run_id *args:
    uv run --locked -m src.tiny_shakespeare.gpt.generate {{run_id}} {{args}}
