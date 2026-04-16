'''
a) Sample 10 documents from TinyStories. Using your previously-trained TinyStories
tokenizer, encode these sampled documents into integer IDs. What is the tokenizer’s
compression ratio (bytes/token)?
(b) Using your TinyStories tokenizer, encode the respective training and development
datasets into a sequence of integer token IDs. We’ll use this later to train our lan-
guage model. We recommend serializing the token IDs as a NumPy array of datatype
uint16. Why is uint16 an appropriate choice


'''
from __future__ import annotations

import json
from typing import Iterable, Iterator

import re
import random
import random
import numpy as np
from pathlib import Path

from eecs148b_hw1.tokenizer import Tokenizer


# -----------------------------
# Paths
# -----------------------------
TRAIN_PATH = "/home/anahita/Spring_2026/CS148b/hw1/data/TinyStoriesV2-GPT4-train.txt"
VAL_PATH = "/home/anahita/Spring_2026/CS148b/hw1/data/TinyStoriesV2-GPT4-valid.txt"

VOCAB_PATH = "bpe_tinystories_10k/vocab.json"
MERGES_PATH = "bpe_tinystories_10k/merges.json"

OUT_DIR = Path("tokenized_tinystories")
OUT_DIR.mkdir(parents=True, exist_ok=True)


SPECIAL_TOKENS = ["<|endoftext|>"]

# -----------------------------
# tokenizer
# -----------------------------
tokenizer = Tokenizer.from_files(
    vocab_filepath=VOCAB_PATH,
    merges_filepath=MERGES_PATH,
    special_tokens=SPECIAL_TOKENS,
)


def load_10_docs_from_tinystories(text_path, seed=0):
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()

    docs = [d for d in text.split("<|endoftext|>") if d.strip()]
    random.seed(seed)
    return random.sample(docs, 10)


# -----------------------------
# Part (a): sample 10 docs + compression ratio
# -----------------------------
def compute_compression_ratio():
    sampled_docs = load_10_docs_from_tinystories(TRAIN_PATH, seed=42)

    total_bytes = 0
    total_tokens = 0

    for doc in sampled_docs:
        token_ids = tokenizer.encode(doc)

        total_bytes += len(doc.encode("utf-8"))
        total_tokens += len(token_ids)

    compression_ratio = total_bytes / total_tokens

    print("----- Compression Ratio -----")
    print("Sampled docs:", len(sampled_docs))
    print("Total bytes:", total_bytes)
    print("Total tokens:", total_tokens)
    print("Compression ratio (bytes/token):", compression_ratio)


# -----------------------------
# encode file to uint16
# -----------------------------
# def encode_file_to_uint16(input_path, output_path):
#     print(f"Encoding: {input_path}")

#     ids = []

#     with open(input_path, "r", encoding="utf-8") as f:
#         for token_id in tokenizer.encode_iterable(f):
#             ids.append(token_id)

#     arr = np.array(ids, dtype=np.uint16)

#     np.save(output_path, arr)

#     print(f"Saved {len(arr)} token ids")
#     print(f"dtype: {arr.dtype}")
#     print(f"output: {output_path}")
def encode_file_to_uint16(input_path, output_path):
    print(f"Encoding: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    docs = text.split("<|endoftext|>")
    all_ids = []

    for i, doc in enumerate(docs):
        if doc:
            all_ids.extend(tokenizer.encode(doc))
            all_ids.append(tokenizer.token_to_id[b"<|endoftext|>"])

        if i % 1000 == 0:
            print(f"Processed {i} documents")

    arr = np.array(all_ids, dtype=np.uint16)
    np.save(output_path, arr)

    print(f"Saved {len(arr)} token ids")
    print(f"dtype: {arr.dtype}")
    print(f"output: {output_path}")   
    
# -----------------------------
# Part (b): encode train/valid
# -----------------------------
def encode_train_val():
    encode_file_to_uint16(
        TRAIN_PATH,
        OUT_DIR / "tinystories_train_ids.npy"
    )

    encode_file_to_uint16(
        VAL_PATH,
        OUT_DIR / "tinystories_valid_ids.npy"
    )



# -----------------------------
# Inspect Longest token 
# -----------------------------
def inspect_longest_token():
    longest_id, longest_token = max(
        tokenizer.vocab.items(),
        key=lambda kv: len(kv[1])
    )

    print("----- Longest Token -----")
    print("Token ID:", longest_id)
    print("Length:", len(longest_token))
    print("Bytes:", longest_token)

    try:
        print("Decoded:", longest_token.decode("utf-8"))
    except UnicodeDecodeError:
        print("Decoded: <invalid standalone utf-8>")
        
        

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    compute_compression_ratio()
    encode_train_val()
    inspect_longest_token()
    
    
'''

Saved 541229348 token ids
dtype: uint16
output: tokenized_tinystories/tinystories_train_ids.npy
Encoding: /home/anahita/Spring_2026/CS148b/hw1/data/TinyStoriesV2-GPT4-valid.txt
Saved 5465884 token ids
dtype: uint16
output: tokenized_tinystories/tinystories_valid_ids.npy
----- Longest Token -----
Token ID: 7160
Length: 15
Bytes: b' accomplishment'
Decoded:  accomplishment
'''