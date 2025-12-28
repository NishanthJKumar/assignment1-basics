from __future__ import annotations
import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Iterator

import regex as re


# Exact pattern from the assignment handout (GPT-2 pre-tokenizer).
PRE_TOKENIZER_REGEX = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _split_keep_delims(text: str, delimiters: list[str]) -> list[str]:
    """
    Split `text` on any of the `delimiters`, keeping the delimiters in the output.
    Uses re.escape on each delimiter so regex metacharacters (e.g. '|') are treated literally.
    Longer delimiters are matched first to handle overlapping special tokens.
    """
    if not delimiters:
        return [text]
    # Sort by length descending so longer tokens match first.
    sorted_delims = sorted(delimiters, key=len, reverse=True)
    escaped = [re.escape(tok) for tok in sorted_delims]
    pattern = "(" + "|".join(escaped) + ")"  # capture group => delimiters are kept
    return re.split(pattern, text)

def pretokenize_from_file_naive(filepath: str, special_tokens: list[str]) -> dict[bytes, int]:
    """
    Correctness-first pretokenization for byte-level BPE training.

    - Reads the entire file as UTF-8 text.
    - Splits on special tokens (keeping them, but NOT pretokenizing them).
    - Runs the GPT-2 regex pre-tokenizer (PRE_TOKENIZER_REGEX) on each non-special span.
    - Returns counts keyed by UTF-8 encoded bytes of each pre-token.
    """
    counts: dict[bytes, int] = {}

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    return pretokenize_naive(text, special_tokens)

def pretokenize_naive(text: str, special_tokens: list[str]) -> dict[bytes, int]:
    """
    Correctness-first pretokenization for byte-level BPE training.

    - Reads the entire file as UTF-8 text.
    - Splits on special tokens (keeping them, but NOT pretokenizing them).
    - Runs the GPT-2 regex pre-tokenizer (PRE_TOKENIZER_REGEX) on each non-special span.
    - Returns counts keyed by UTF-8 encoded bytes of each pre-token.
    """
    counts: dict[bytes, int] = {}

    parts = _split_keep_delims(text, special_tokens)

    for part in parts:
        if part in special_tokens:
            continue

        for m in re.finditer(PRE_TOKENIZER_REGEX, part):
            tok_bytes = m.group(0).encode("utf-8")
            counts[tok_bytes] = counts.get(tok_bytes, 0) + 1

    return counts


# ---- NAIVE BPE TRAINING (minimal changes, but correct) ----

def _bytes_to_symbols(b: bytes) -> tuple[bytes, ...]:
    # Represent as tuple of byte-symbols, each symbol is a bytes of length 1.
    return tuple(bytes([x]) for x in b)


def compute_frequency_pairs(curr_tokens_dict: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, bytes], int]:
    """Count adjacent symbol-pair frequencies, weighted by token frequency."""
    pair_counts: dict[tuple[bytes, bytes], int] = {}
    for sym_seq, freq in curr_tokens_dict.items():
        if freq <= 0 or len(sym_seq) < 2:
            continue
        for i in range(len(sym_seq) - 1):
            pair = (sym_seq[i], sym_seq[i + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + freq
    return pair_counts


def find_most_frequent_pair(frequency_pairs_dict: dict[tuple[bytes, bytes], int]) -> tuple[bytes, bytes] | None:
    """
    Return the most frequent pair; break ties by lexicographically greater pair
    (per assignment spec).
    """
    if not frequency_pairs_dict:
        return None
    # max by (count, pair) gives lexicographically greatest pair on ties
    return max(frequency_pairs_dict.items(), key=lambda kv: (kv[1], kv[0]))[0]


def compute_updated_freqs_dict(
    curr_tokens_dict: dict[tuple[bytes, ...], int],
    most_freq_pair: tuple[bytes, bytes],
) -> dict[tuple[bytes, ...], int]:
    """Rewrite all symbol sequences by merging every occurrence of most_freq_pair."""
    a, b = most_freq_pair
    merged = a + b

    new_dict: dict[tuple[bytes, ...], int] = {}
    for sym_seq, freq in curr_tokens_dict.items():
        if len(sym_seq) < 2:
            new_dict[sym_seq] = new_dict.get(sym_seq, 0) + freq
            continue

        out: list[bytes] = []
        i = 0
        while i < len(sym_seq):
            if i < len(sym_seq) - 1 and sym_seq[i] == a and sym_seq[i + 1] == b:
                out.append(merged)
                i += 2
            else:
                out.append(sym_seq[i])
                i += 1

        out_tup = tuple(out)
        new_dict[out_tup] = new_dict.get(out_tup, 0) + freq

    return new_dict


def bpe_tokenize_naive_after_pretokenize(
    pretokenized_dict: dict[bytes, int],
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train BPE merges naively from a pretokenized frequency table.

    Returns:
      vocab: dict[int, bytes]  (special tokens, 256 bytes, then merged tokens)
      merges: list[tuple[bytes, bytes]]  in creation order
    """
    base_vocab_size = 256 + len(special_tokens)
    if vocab_size < base_vocab_size:
        raise ValueError(f"vocab_size must be >= 256 + len(special_tokens) = {base_vocab_size}")

    max_merges = vocab_size - base_vocab_size

    # Convert pretokenized bytes -> tuple-of-symbols (bytes) representation.
    curr_tokens_dict: dict[tuple[bytes, ...], int] = {}
    for tok_bytes, freq in pretokenized_dict.items():
        sym_seq = _bytes_to_symbols(tok_bytes)
        curr_tokens_dict[sym_seq] = curr_tokens_dict.get(sym_seq, 0) + freq

    merge_history: list[tuple[bytes, bytes]] = []

    while len(merge_history) < max_merges:
        freq_pairs_dict = compute_frequency_pairs(curr_tokens_dict)
        most_freq_pair = find_most_frequent_pair(freq_pairs_dict)
        if most_freq_pair is None:
            break

        merge_history.append(most_freq_pair)
        curr_tokens_dict = compute_updated_freqs_dict(curr_tokens_dict, most_freq_pair)

    # Build final vocab: special tokens first, then 256 single-byte tokens, then merged tokens.
    vocab: dict[int, bytes] = {}
    next_id = 0

    for st in special_tokens:
        vocab[next_id] = st.encode("utf-8")
        next_id += 1

    for b in range(256):
        vocab[next_id] = bytes([b])
        next_id += 1

    for a, b in merge_history:
        vocab[next_id] = a + b
        next_id += 1

    return vocab, merge_history


def tokenize_bpe_naive(
    filepath: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Naive, correctness-first end-to-end BPE training:
      1) pretokenize
      2) train merges until vocab budget
      3) return (vocab, merges)
    """
    pretokenized_counts = pretokenize_from_file_naive(filepath, special_tokens)
    return bpe_tokenize_naive_after_pretokenize(pretokenized_counts, vocab_size, special_tokens)



@dataclass
class Tokenizer:
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]
    special_tokens: list[str] | None = None

    def __post_init__(self):
        # Build reverse vocab: bytes -> int for fast encoding.
        self._bytes_to_id: dict[bytes, int] = {v: k for k, v in self.vocab.items()}
        # Build merge rank: merged_bytes -> priority (lower = earlier merge).
        self._merge_rank: dict[tuple[bytes, bytes], int] = {
            pair: i for i, pair in enumerate(self.merges)
        }

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None) -> Tokenizer:
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab_raw = json.load(f)
        # vocab_raw is str -> int, we need int -> bytes
        vocab: dict[int, bytes] = {}
        for tok_str, tok_id in vocab_raw.items():
            vocab[tok_id] = tok_str.encode("utf-8")

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line or " " not in line:
                    continue
                parts = line.split(" ")
                if len(parts) == 2:
                    merges.append((parts[0].encode("utf-8"), parts[1].encode("utf-8")))

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def _apply_merges(self, token_bytes: bytes) -> list[bytes]:
        """Apply BPE merges to a single pre-token, returning list of merged byte sequences."""
        if len(token_bytes) == 0:
            return []
        # Start with individual bytes.
        symbols: list[bytes] = [bytes([b]) for b in token_bytes]
        while len(symbols) > 1:
            # Find the pair with the smallest merge rank (earliest merge).
            best_pair = None
            best_rank = None
            best_idx = None
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                if pair in self._merge_rank:
                    rank = self._merge_rank[pair]
                    if best_rank is None or rank < best_rank:
                        best_pair = pair
                        best_rank = rank
                        best_idx = i
            if best_pair is None:
                break
            # Merge at best_idx.
            merged = best_pair[0] + best_pair[1]
            symbols = symbols[:best_idx] + [merged] + symbols[best_idx + 2:]
        return symbols

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        special = self.special_tokens or []
        parts = _split_keep_delims(text, special)
        ids: list[int] = []
        for part in parts:
            if not part:
                continue
            if part in special:
                # Special token: encode directly.
                ids.append(self._bytes_to_id[part.encode("utf-8")])
            else:
                # Apply GPT-2 pre-tokenizer, then BPE merges.
                for m in re.finditer(PRE_TOKENIZER_REGEX, part):
                    tok_bytes = m.group(0).encode("utf-8")
                    merged_symbols = self._apply_merges(tok_bytes)
                    for sym in merged_symbols:
                        ids.append(self._bytes_to_id[sym])
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Streaming version of encode, yields token IDs one at a time."""
        special = self.special_tokens or []
        for chunk in iterable:
            parts = _split_keep_delims(chunk, special)
            for part in parts:
                if not part:
                    continue
                if part in special:
                    yield self._bytes_to_id[part.encode("utf-8")]
                else:
                    for m in re.finditer(PRE_TOKENIZER_REGEX, part):
                        tok_bytes = m.group(0).encode("utf-8")
                        merged_symbols = self._apply_merges(tok_bytes)
                        for sym in merged_symbols:
                            yield self._bytes_to_id[sym]

    def decode(self, ids: list[int]) -> str:
        b_to_decode = b""
        for token_id in ids:
            b_to_decode += self.vocab[token_id]
        return b_to_decode.decode("utf-8", errors="replace")


# Simple tokenization example.
if __name__ == "__main__":
    SPECIAL_TOKENS = ["<|endoftext|>"]
    vocab, merges = tokenize_bpe_naive("data/njk_mini_example.txt", vocab_size=265, special_tokens=SPECIAL_TOKENS)
    print(len(vocab), len(merges))
