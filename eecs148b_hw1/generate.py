import argparse
import torch

from eecs148b_hw1.transformer_lm import TransformerLM
from eecs148b_hw1.tokenizer import Tokenizer
from eecs148b_hw1.decoding import decode


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--vocab-path", type=str, required=True)
    parser.add_argument("--merges-path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


def main():
    args = parse_args()

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    config = checkpoint["args"]
    state_dict = checkpoint["model_state_dict"]

    tokenizer = Tokenizer.from_files(
        vocab_filepath=args.vocab_path,
        merges_filepath=args.merges_path,
        special_tokens=["<|endoftext|>"],
    )

    model = TransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        device=args.device,
        dtype=torch.float32,
    ).to(args.device)

    model.load_state_dict(state_dict)
    model.eval()

    prompt_ids = tokenizer.encode(args.prompt)
    eos_token_id = tokenizer.token_to_id.get(b"<|endoftext|>")

    output_ids = decode(
        model=model,
        prompt=prompt_ids,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=eos_token_id,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    output_text = tokenizer.decode(output_ids)

    print("========== PROMPT ==========")
    print(args.prompt)
    print()
    print("========== GENERATED TEXT ==========")
    print(output_text)


if __name__ == "__main__":
    main()