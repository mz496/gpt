run:
    uv run --locked -m src.tiny_shakespeare.main

test *args:
    uv run --locked pytest {{args}}

bigram *args:
    uv run --locked -m src.tiny_shakespeare.bigram {{args}}
