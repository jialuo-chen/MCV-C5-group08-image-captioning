from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Sequence
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


class BaseTokenizer(ABC):
    """Common tokenizer interface."""

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Encode text to a list of token ids (including SOS/EOS)."""

    @abstractmethod
    def decode(self, ids: Sequence[int]) -> str:
        """Decode token ids back to text (stripping SOS/EOS/PAD)."""

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...

    @property
    @abstractmethod
    def pad_id(self) -> int: ...

    @property
    @abstractmethod
    def sos_id(self) -> int: ...

    @property
    @abstractmethod
    def eos_id(self) -> int: ...

    def pad_sequence(self, ids: list[int], max_length: int) -> list[int]:
        """Pad or truncate *ids* to *max_length*."""
        if len(ids) >= max_length:
            return ids[:max_length]
        return ids + [self.pad_id] * (max_length - len(ids))

    def save(self, path: str | Path) -> None:
        """Persist tokenizer state to *path* (JSON)."""
        raise NotImplementedError

    @classmethod
    def load(cls, path: str | Path) -> "BaseTokenizer":
        """Load tokenizer state from *path*."""
        raise NotImplementedError


_DEFAULT_CHARS = [
    "<SOS>", "<EOS>", "<PAD>",
    " ", "!", '"', "#", "&", "'", "(", ")", ",", "-", ".", "0", "1", "2",
    "3", "4", "5", "6", "7", "8", "9", ":", ";", "=", "?",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
]


class CharTokenizer(BaseTokenizer):
    """Character-level tokenizer with fixed vocabulary."""

    def __init__(self, chars: list[str] | None = None) -> None:
        self._chars = chars if chars is not None else list(_DEFAULT_CHARS)
        self._idx2char = {i: c for i, c in enumerate(self._chars)}
        self._char2idx = {c: i for i, c in enumerate(self._chars)}
        self._unk_char = " "  # map unknown chars to space

    def encode(self, text: str) -> list[int]:
        ids = [self._char2idx["<SOS>"]]
        for ch in text:
            ids.append(self._char2idx.get(ch, self._char2idx[self._unk_char]))
        ids.append(self._char2idx["<EOS>"])
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        chars: list[str] = []
        for i in ids:
            tok = self._idx2char.get(i, "")
            if tok in ("<SOS>", "<PAD>"):
                continue
            if tok == "<EOS>":
                break
            chars.append(tok)
        return "".join(chars)

    @property
    def vocab_size(self) -> int:
        return len(self._chars)

    @property
    def pad_id(self) -> int:
        return self._char2idx["<PAD>"]

    @property
    def sos_id(self) -> int:
        return self._char2idx["<SOS>"]

    @property
    def eos_id(self) -> int:
        return self._char2idx["<EOS>"]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"type": "char", "chars": self._chars}
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        data = json.loads(Path(path).read_text())
        return cls(chars=data["chars"])

class WordTokenizer(BaseTokenizer):
    """Whitespace-split word-level tokenizer.

    Build vocabulary from a corpus of captions via ``build_vocab``.
    """

    SPECIAL_TOKENS = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]

    def __init__(self, vocab: list[str] | None = None) -> None:
        if vocab is not None:
            self._vocab = list(vocab)
        else:
            self._vocab = list(self.SPECIAL_TOKENS)
        self._rebuild_maps()

    def _rebuild_maps(self) -> None:
        self._idx2word = {i: w for i, w in enumerate(self._vocab)}
        self._word2idx = {w: i for i, w in enumerate(self._vocab)}

    def build_vocab(self, captions: list[str], min_freq: int = 2, max_vocab: int | None = None) -> None:
        """Build vocabulary from *captions*."""
        counter: Counter[str] = Counter()
        for cap in captions:
            counter.update(self._tokenize(cap))
        words = [w for w, c in counter.most_common() if c >= min_freq]
        if max_vocab is not None:
            words = words[:max_vocab - len(self.SPECIAL_TOKENS)]
        self._vocab = list(self.SPECIAL_TOKENS) + words
        self._rebuild_maps()

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        text = text.lower().strip()
        tokens = re.findall(r"\w+|[^\w\s]", text)
        return tokens

    def encode(self, text: str) -> list[int]:
        unk = self._word2idx["<UNK>"]
        ids = [self._word2idx["<SOS>"]]
        for w in self._tokenize(text):
            ids.append(self._word2idx.get(w, unk))
        ids.append(self._word2idx["<EOS>"])
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        words: list[str] = []
        for i in ids:
            tok = self._idx2word.get(i, "")
            if tok in ("<SOS>", "<PAD>"):
                continue
            if tok == "<EOS>":
                break
            words.append(tok)
        return " ".join(words)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @property
    def pad_id(self) -> int:
        return self._word2idx["<PAD>"]

    @property
    def sos_id(self) -> int:
        return self._word2idx["<SOS>"]

    @property
    def eos_id(self) -> int:
        return self._word2idx["<EOS>"]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"type": "word", "vocab": self._vocab}
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "WordTokenizer":
        data = json.loads(Path(path).read_text())
        return cls(vocab=data["vocab"])


class SubwordTokenizer(BaseTokenizer):
    """BPE subword tokenizer backed by HuggingFace ``tokenizers``.

    Train a new BPE model via ``train`` or load an existing one.
    """

    def __init__(self) -> None:
        self._tokenizer = None  # lazy init
        self._pad_id = 0
        self._sos_id = 1
        self._eos_id = 2

    def train(self, captions: list[str], vocab_size: int = 4000) -> None:
        """Train a BPE tokenizer on *captions*."""

        tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<PAD>", "<SOS>", "<EOS>", "<UNK>"],
        )
        tokenizer.train_from_iterator(captions, trainer=trainer)
        self._tokenizer = tokenizer
        self._pad_id = tokenizer.token_to_id("<PAD>")
        self._sos_id = tokenizer.token_to_id("<SOS>")
        self._eos_id = tokenizer.token_to_id("<EOS>")

    def encode(self, text: str) -> list[int]:
        enc = self._tokenizer.encode(text)
        return [self._sos_id] + enc.ids + [self._eos_id]

    def decode(self, ids: Sequence[int]) -> str:
        filtered = []
        for i in ids:
            if i == self._sos_id or i == self._pad_id:
                continue
            if i == self._eos_id:
                break
            filtered.append(i)
        return self._tokenizer.decode(filtered)

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    @property
    def pad_id(self) -> int:
        return self._pad_id

    @property
    def sos_id(self) -> int:
        return self._sos_id

    @property
    def eos_id(self) -> int:
        return self._eos_id

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._tokenizer.save(str(path))

    @classmethod
    def load(cls, path: str | Path) -> "SubwordTokenizer":
        tok = cls()
        tok._tokenizer = Tokenizer.from_file(str(path))
        tok._pad_id = tok._tokenizer.token_to_id("<PAD>")
        tok._sos_id = tok._tokenizer.token_to_id("<SOS>")
        tok._eos_id = tok._tokenizer.token_to_id("<EOS>")
        return tok


def build_tokenizer(cfg, captions: list[str] | None = None) -> BaseTokenizer:
    """Build a tokenizer from config.

    Parameters
    ----------
    cfg : Config
        Tokenizer section of the experiment config.
    captions : list[str] | None
        Training captions (needed for word/subword tokenizers to build vocab).
    """
    tok_type = cfg["type"]
    if tok_type == "char":
        return CharTokenizer()
    elif tok_type == "word":
        tok = WordTokenizer()
        if captions:
            tok.build_vocab(
                captions,
                min_freq=cfg.get("min_freq", 2),
                max_vocab=cfg.get("vocab_size"),
            )
        return tok
    elif tok_type == "subword":
        tok = SubwordTokenizer()
        if captions:
            tok.train(captions, vocab_size=cfg.get("vocab_size", 4000))
        return tok
    else:
        raise ValueError(f"Unknown tokenizer type: {tok_type}")
