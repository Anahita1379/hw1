import torch

from eecs148b_hw1.softmax import softmax


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """
    logits: (batch_size, vocab_size)
    returns: (batch_size,) sampled token ids
    """
    
    if temperature < 0:
        raise ValueError("temperature must be nonnegative")
    if not (0 < top_p <= 1.0):
        raise ValueError("top_p must be in (0, 1]")

    # tau = 0 => greedy decoding
    if temperature == 0:
        return torch.argmax(logits, dim=-1)

    scaled_logits = logits / temperature
    probs = softmax(scaled_logits, dim=-1)

    if top_p < 1.0:
        vocab_size = probs.shape[-1]

        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Keep the smallest prefix whose cumulative probability reaches top_p
        keep_sorted = cumulative_probs <= top_p
        keep_sorted[..., 0] = True  # always keep at least one token

        # Also keep the first token that crosses the threshold
        shifted_keep = keep_sorted.clone()
        shifted_keep[..., 1:] = keep_sorted[..., :-1]
        shifted_keep[..., 0] = True
        keep_sorted = shifted_keep

        filtered_sorted_probs = torch.where(
            keep_sorted,
            sorted_probs,
            torch.zeros_like(sorted_probs),
        )

        # Renormalize
        filtered_sorted_probs = filtered_sorted_probs / filtered_sorted_probs.sum(dim=-1, keepdim=True)

        # Sample in sorted space
        sampled_sorted_idx = torch.multinomial(filtered_sorted_probs, num_samples=1).squeeze(-1)

        # Map back to original vocab ids
        next_token = sorted_indices.gather(
            dim=-1,
            index=sampled_sorted_idx.unsqueeze(-1)
        ).squeeze(-1)
        return next_token

    return torch.multinomial(probs, num_samples=1).squeeze(-1)


@torch.no_grad()
def decode(
    model,
    prompt: list[int],
    max_new_tokens: int,
    eos_token_id: int | None = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> list[int]:
    """
    Generate a single completion from a single prompt.

    Args:
        model: TransformerLM
        prompt: list of token ids
        max_new_tokens: maximum number of tokens to generate
        eos_token_id: if generated, stop decoding
        temperature: softmax temperature
        top_p: nucleus sampling threshold

    Returns:
        Full sequence = prompt + generated tokens
    """
    
    
    device = next(model.parameters()).device
    model.eval()

    tokens = torch.tensor(prompt, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)

    for _ in range(max_new_tokens):
        # Respect context window if prompt grows too long
        if tokens.shape[1] > model.context_length:
            input_tokens = tokens[:, -model.context_length:]
        else:
            input_tokens = tokens

        logits = model(input_tokens)              # (1, seq_len, vocab_size)
        next_logits = logits[:, -1, :]            # (1, vocab_size)

        next_token = sample_next_token(
            next_logits,
            temperature=temperature,
            top_p=top_p,
        )                                         # (1,)

        tokens = torch.cat([tokens, next_token.unsqueeze(1)], dim=1)

        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    return tokens.squeeze(0).tolist()


@torch.no_grad()
def decode_batch(
    model,
    prompts: list[list[int]],
    max_new_tokens: int,
    eos_token_id: int | None = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
    pad_token_id: int = 0,
) -> list[list[int]]:
    """
    Batched decoding for prompts of different lengths.
    Simpler than a production implementation, but useful.

    Returns one full generated sequence per prompt.
    """
    device = next(model.parameters()).device
    model.eval()

    batch_size = len(prompts)
    sequences = [list(p) for p in prompts]
    finished = [False] * batch_size

    for _ in range(max_new_tokens):
        current_lengths = [len(s) for s in sequences]
        max_len = max(current_lengths)

        # Clip to context length
        max_len = min(max_len, model.context_length)

        batch = []
        for seq in sequences:
            clipped = seq[-max_len:]
            padded = [pad_token_id] * (max_len - len(clipped)) + clipped
            batch.append(padded)

        x = torch.tensor(batch, dtype=torch.long, device=device)  # (B, T)
        logits = model(x)                                         # (B, T, V)

        next_logits_list = []
        for i, seq in enumerate(sequences):
            effective_len = min(len(seq), model.context_length)
            next_logits_list.append(logits[i, effective_len - 1, :])

        next_logits = torch.stack(next_logits_list, dim=0)        # (B, V)
        next_tokens = sample_next_token(
            next_logits,
            temperature=temperature,
            top_p=top_p,
        )                                                         # (B,)

        for i in range(batch_size):
            if finished[i]:
                continue
            nxt = next_tokens[i].item()
            sequences[i].append(nxt)
            if eos_token_id is not None and nxt == eos_token_id:
                finished[i] = True

        if all(finished):
            break

    return sequences