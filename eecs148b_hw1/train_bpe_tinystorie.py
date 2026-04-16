'''
Train a byte-level BPE tokenizer on the TinyStories dataset, using a maximum vocabulary
size of 10,000. Make sure to add the TinyStories <|endoftext|> special token to the vocab-
ulary. Serialize the resulting vocabulary and merges to disk for further inspection. What is
the longest token in the vocabulary? Does it make sense
'''




import json
from pathlib import Path

from eecs148b_hw1.train_bpe import train_bpe
import data

def main():
    input_path = "/home/anahita/Spring_2026/CS148b/hw1/data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]

    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )

    out_dir = Path("bpe_tinystories_10k")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save vocab as token_id -> utf-8-safe representation
    vocab_json = {}
    for token_id, token_bytes in vocab.items():
        vocab_json[token_id] = list(token_bytes)

    with open(out_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_json, f)

    # Save merges in order
    merges_json = [
        [list(left), list(right)]
        for (left, right) in merges
    ]
    with open(out_dir / "merges.json", "w", encoding="utf-8") as f:
        json.dump(merges_json, f)

    # Find longest token
    longest_token_id, longest_token_bytes = max(
        vocab.items(),
        key=lambda kv: len(kv[1])
    )

    print("Vocab size:", len(vocab))
    print("Num merges:", len(merges))
    print("Longest token id:", longest_token_id)
    print("Longest token length:", len(longest_token_bytes))
    print("Longest token bytes:", longest_token_bytes)

    try:
        print("Longest token decoded:", longest_token_bytes.decode("utf-8"))
    except UnicodeDecodeError:
        print("Longest token decoded: <not valid standalone utf-8>")


if __name__ == "__main__":
    main()
    
    
'''
Vocab size: 10000
Num merges: 9743
Longest token id: 7160
Longest token length: 15
Longest token bytes: b' accomplishment'
Longest token decoded:  accomplishment
'''
