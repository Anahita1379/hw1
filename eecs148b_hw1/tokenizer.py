
from __future__ import annotations

import json
from typing import Iterable, Iterator
import regex as re


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    
    def __init__(self, vocab, merges, special_tokens=None):


        '''
        Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens. This function
        should accept the following parameters:
        • vocab: dict[int, bytes]
        • merges: list[tuple[bytes, bytes]]
        • special tokens: list[str] | None = None
        '''
        self.vocab: dict[int, bytes] = dict(vocab)
        self.merges: list[tuple[bytes, bytes]] = list(merges)
        self.special_tokens = special_tokens or []
        
        # token bytes -> token id
        self.token_to_id: dict[bytes, int] = {token: idx for idx, token in self.vocab.items()}
        
        # Add user-provided special tokens if missing
        next_id = max(self.vocab.keys()) + 1 if self.vocab else 0
        for tok in self.special_tokens:
            b = tok.encode("utf-8")
            if b not in self.token_to_id:
                self.vocab[next_id] = b
                self.token_to_id[b] = next_id
                next_id += 1
                
        # merge ranks for greedy BPE
        self.merge_ranks: dict[tuple[bytes, bytes], int] = {
            pair: i for i, pair in enumerate(self.merges)
        }
        
        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        self.special_tokens_sorted = sorted_special_tokens
        
        if self.special_tokens:
            self.special_pattern = re.compile(
                "(" + "|".join(re.escape(tok) for tok in self.special_tokens_sorted) + ")"
            )
        else:
            self.special_pattern = None



    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        '''
        Class method that constructs and returns a Tokenizer from a serialized vocabulary
        and list of merges (in the same format that your BPE training code output) and
        (optionally) a list of special tokens. This method should accept the following additional
        parameters:
        • vocab filepath: str
        • merges filepath: str
        '''
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)

        with open(merges_filepath, "r", encoding="utf-8") as f:
            raw_merges = json.load(f)

        vocab = {int(k): bytes(v) for k, v in raw_vocab.items()}
        merges = [(bytes(a), bytes(b)) for a, b in raw_merges]

        return cls(vocab, merges, special_tokens=special_tokens)

    def _split_special_tokens(self, text: str) -> list[str]:
        if not self.special_pattern:
            return [text]
        return [part for part in self.special_pattern.split(text) if part != ""]

    def _bpe_encode_pretoken(self, pretoken: str) -> list[bytes]:
        # start from bytes
        parts = [bytes([b]) for b in pretoken.encode("utf-8")]

        if len(parts) <= 1:
            return parts

        while True:
            best_pair = None
            best_rank = None

            for i in range(len(parts) - 1):
                pair = (parts[i], parts[i + 1])
                rank = self.merge_ranks.get(pair)
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            # merge all occurrences of the selected pair
            new_parts = []
            i = 0
            while i < len(parts):
                if (
                    i < len(parts) - 1
                    and parts[i] == best_pair[0]
                    and parts[i + 1] == best_pair[1]
                ):
                    new_parts.append(parts[i] + parts[i + 1])
                    i += 2
                else:
                    new_parts.append(parts[i])
                    i += 1
            parts = new_parts

        return parts


    def encode(self, text: str) -> list[int]:
        '''
        Encode an input text into a sequence of
        token IDs.
        '''
        ids = []

        for chunk in self._split_special_tokens(text):
            if chunk in self.special_tokens:
                ids.append(self.token_to_id[chunk.encode("utf-8")])
                continue

            for match in re.finditer(PAT, chunk):
                pretoken = match.group(0)
                bpe_tokens = self._bpe_encode_pretoken(pretoken)
                for tok in bpe_tokens:
                    ids.append(self.token_to_id[tok])

        return ids


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        '''
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields
        token IDs. This supports tokenizing large inputs without materializing the full token
        sequence in memory.
        '''
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id
    

    def decode(self, ids: list[int]) -> str:
        '''
        Decode a sequence of token IDs into text.
        '''
        out = bytearray()
        for idx in ids:
            token_bytes = self.vocab[idx]
            out.extend(token_bytes)
        return bytes(out).decode("utf-8", errors="replace")
    