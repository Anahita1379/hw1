from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
# from typing import Dict, List, Tuple
import regex as re
import os
from typing import Any


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""



def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str] | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    
    special_tokens = special_tokens or []
    
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
        
    # Initial byte vocabulary.
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_id = 256
    
    # Add special tokens to vocabulary with fixed IDs after the byte vocab.
    for tok in special_tokens:
        b = tok.encode("utf-8")
        if b not in vocab.values():
            vocab[next_id] = b
            next_id += 1
            
    # If special tokens already exhaust the budget, stop early.
    if len(vocab) >= vocab_size:
        return vocab, [] 
    
    # Split on special tokens so no merges cross them.
    if special_tokens:
        split_pat = "|".join(re.escape(tok) for tok in special_tokens)
        segments = re.split(split_pat, text)
    else:
        segments = [text]
        
    # Count pre-tokens without storing every occurrence.
    pretoken_counts: Counter[tuple[bytes, ...]] = Counter()
    for segment in segments:
        for m in re.finditer(PAT, segment):
            s = m.group(0)
            btok = tuple(bytes([b]) for b in s.encode("utf-8"))
            if btok:
                pretoken_counts[btok] += 1
                
    # Working representation: each unique pretoken is a tuple of token-bytes.
    # Example: (b'h', b'e', b'l', b'l', b'o')
    token_seqs = dict(pretoken_counts)
    
    # Pair -> count over the weighted corpus.
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    
    # Pair -> set of token sequences containing that pair.
    pair_to_words: defaultdict[tuple[bytes, bytes], set] = defaultdict(set)
    
    def add_word_pairs(word: tuple[bytes, ...], freq: int) -> None:
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] += freq
            pair_to_words[pair].add(word)
            
    for word, freq in token_seqs.items():
        add_word_pairs(word, freq)

    merges: list[tuple[bytes, bytes]] = []
    target_num_merges = vocab_size - len(vocab)

    for _ in range(target_num_merges):
        if not pair_counts:
            break

        max_freq = max(pair_counts.values())
        if max_freq <= 0:
            break

        # Deterministic tie-break: lexicographically greatest pair among max-frequency pairs.
        best_pair = max(pair for pair, c in pair_counts.items() if c == max_freq)
        a, b = best_pair
        merged = a + b

        merges.append(best_pair)
        vocab[next_id] = merged
        next_id += 1

        affected_words = list(pair_to_words.get(best_pair, set()))
        if not affected_words:
            continue

        for old_word in affected_words:
            freq = token_seqs.pop(old_word, 0)
            if freq == 0:
                continue

            # Remove old word's contribution from indices.
            for i in range(len(old_word) - 1):
                p = (old_word[i], old_word[i + 1])
                pair_counts[p] -= freq
                if pair_counts[p] == 0:
                    del pair_counts[p]
                s = pair_to_words.get(p)
                if s is not None:
                    s.discard(old_word)
                    if not s:
                        pair_to_words.pop(p, None)

            # Replace all occurrences of best_pair in the word.
            new_tokens: list[bytes] = []
            i = 0
            while i < len(old_word):
                if i < len(old_word) - 1 and old_word[i] == a and old_word[i + 1] == b:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(old_word[i])
                    i += 1
            new_word = tuple(new_tokens)

            token_seqs[new_word] = token_seqs.get(new_word, 0) + freq

            # Add new word's contribution.
            for i in range(len(new_word) - 1):
                p = (new_word[i], new_word[i + 1])
                pair_counts[p] += freq
                pair_to_words[p].add(new_word)

    return vocab, merges


