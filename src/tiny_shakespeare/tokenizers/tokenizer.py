import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Self

# Bump when the saved layout changes, so old files fail loudly instead of loading wrong.
SAVE_FORMAT = 2


class Tokenizer(ABC):
    """
    Interface for turning text into integer tokens and back.

    A tokenizer can have special tokens (e.g. "<|endoftext|>"): strings that are never learned from or split up, and that always map to one reserved id.
    Register them with register_special_tokens before train(). They get the ids right after everything train() learns, in registration order,
    so no merge can ever involve one and the learned ids don't depend on which special tokens exist.

    Subclasses implement the plain-text hooks (_train_on_pieces, _encode_plain, _decode_plain, ...), which never see a special token.
    """

    # Identifies the kind of tokenizer in saved files, and matches the names make_tokenizer takes.
    name: ClassVar[str]

    def __init__(self):
        self._special_tokens: list[str] = []
        # Matches any of the special tokens (longest first); the capture group makes re.split keep them.
        self._special_pattern: re.Pattern[str] | None = None
        self._vocab_cache: dict[int, bytes] | None = None
        self._trained = False

    def register_special_tokens(self, tokens: list[str]) -> None:
        """
        Sets the special tokens, replacing any registered before (so [] clears them). Must be called before train().

        It's an error afterwards because the special tokens decide how many merges train() may learn (see _train_on_pieces),
        and the vocab of a trained tokenizer shouldn't change.
        """
        if self._trained:
            raise ValueError("special tokens must be registered before train()")
        if any(not isinstance(token, str) or token == "" for token in tokens):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ValueError(f"special tokens must be non-empty strings, got {tokens!r}")
        if len(set(tokens)) != len(tokens):
            raise ValueError(f"special tokens must be unique, got {tokens!r}")
        self._special_tokens = list(tokens)
        # Longest first, so that when one token is a prefix of another the longer one wins.
        alternatives = (re.escape(token) for token in sorted(tokens, key=len, reverse=True))
        self._special_pattern = re.compile(f"({'|'.join(alternatives)})") if tokens else None
        self._vocab_cache = None

    @property
    def special_tokens(self) -> list[str]:
        """The registered special tokens, in registration order."""
        return list(self._special_tokens)

    @property
    def special_token_ids(self) -> dict[str, int]:
        """Each special token's id. Only final once trained: the ids come right after the ones learned so far."""
        first_id = self._num_learned_tokens()
        return {token: first_id + i for i, token in enumerate(self._special_tokens)}

    def train(self, s: str) -> None:
        """Learn the vocab from the given text. The special tokens in it are skipped, so nothing is learned from them or across them."""
        pieces = [piece for piece in self._split_on_special_tokens(s) if piece]
        self._train_on_pieces(pieces)
        self._vocab_cache = None
        self._trained = True

    def encode_string(self, s: str) -> list[int]:
        """Text -> tokens. Every occurrence of a special token becomes its id."""
        if self._special_pattern is None:
            return self._encode_plain(s)
        special_ids = self.special_token_ids
        tokens: list[int] = []
        # With a capture group, re.split alternates plain text and the matched special tokens, starting with plain text.
        # e.g. [asdfSPECIAL1fdsaSPECIAL2qqSPECIAL3] -> [asdf, SPECIAL1, fdsa, SPECIAL2, qq, SPECIAL3, ""]
        for i, part in enumerate(self._special_pattern.split(s)):
            if i % 2 == 1:
                tokens.append(special_ids[part])
            elif part:
                tokens.extend(self._encode_plain(part))
        return tokens

    def decode_as_string(self, tokens: list[int]) -> str:
        """Tokens -> text. The id of a special token becomes its text."""
        if self._special_pattern is None:
            return self._decode_plain(tokens)
        special_by_id = {token_id: token for token, token_id in self.special_token_ids.items()}
        parts: list[str] = []
        plain_run: list[int] = []
        for token in tokens:
            if token in special_by_id:
                if plain_run:
                    parts.append(self._decode_plain(plain_run))
                    plain_run = []
                parts.append(special_by_id[token])
            else:
                plain_run.append(token)
        if plain_run:
            parts.append(self._decode_plain(plain_run))
        return "".join(parts)

    def get_vocab(self) -> dict[int, bytes]:
        """
        Maps every token id to the UTF-8 bytes it decodes to (for a special token, its text). Its length is the vocab size.

        Callers (e.g. training loops) may call this often, so it's cached. Callers must not mutate it.
        Note that before train() is called, the learned part is just the initial vocab (e.g. all raw bytes, unmerged).
        """
        if self._vocab_cache is None:
            vocab = dict(self._learned_vocab())
            for token, token_id in self.special_token_ids.items():
                vocab[token_id] = token.encode("utf-8")
            self._vocab_cache = vocab
        return self._vocab_cache

    def _split_on_special_tokens(self, s: str) -> list[str]:
        """The plain text between the special tokens (which are dropped). Empty strings are possible."""
        if self._special_pattern is None:
            return [s]
        return self._special_pattern.split(s)[::2]

    @abstractmethod
    def _train_on_pieces(self, pieces: list[str]) -> None:
        """
        Learn from the plain text pieces between special tokens (never empty, possibly none). Nothing may be learned across two pieces.
        A tokenizer with a target vocab size must leave room in it for the special tokens: len(self._special_tokens) slots.
        """

    @abstractmethod
    def _encode_plain(self, s: str) -> list[int]:
        """Text without special tokens -> tokens."""

    @abstractmethod
    def _decode_plain(self, tokens: list[int]) -> str:
        """Tokens that aren't special tokens -> text."""

    @abstractmethod
    def _num_learned_tokens(self) -> int:
        """How many ids the learned (non-special) vocab uses; the special tokens' ids start right after."""

    @abstractmethod
    def _learned_vocab(self) -> dict[int, bytes]:
        """Maps each learned (non-special) id to the bytes it decodes to."""

    @abstractmethod
    def state(self) -> dict[str, Any]:
        """Everything learned in train() apart from the special tokens, as JSON-serializable values, so that from_state can rebuild an identical tokenizer."""

    @classmethod
    @abstractmethod
    def from_state(cls, state: dict[str, Any]) -> Self:
        """
        The inverse of save(): a trained tokenizer, without needing the training text.
        `state` is the whole saved file, i.e. state() plus "special_tokens".
        """

    def save(self, path: Path) -> None:
        """Writes the trained tokenizer to a JSON file; load it back with load_tokenizer (in tokenizers/utils.py)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"format": SAVE_FORMAT, "name": self.name, "special_tokens": self._special_tokens, **self.state()}))
