MAX_LEN = 512
HEAD_LEN = 128
TAIL_LEN = 382


def head_tail_tokenize(text, tokenizer, max_len=MAX_LEN, head_len=HEAD_LEN, tail_len=TAIL_LEN):
    """
    Tokenize text with head+tail truncation for long sequences.

    If the tokenized text (without special tokens) exceeds (max_len - 2) tokens,
    we keep the first `head_len` tokens and the last `tail_len` tokens,
    then wrap with [CLS] and [SEP].

    For shorter texts, standard truncation and padding is applied.
    """
    # Tokenize without special tokens first to get raw token IDs
    tokens = tokenizer.encode(text, add_special_tokens=False)
    content_max = max_len - 2  # Reserve space for [CLS] and [SEP]

    if len(tokens) > content_max:
        # Head+Tail truncation
        head_tokens = tokens[:head_len]
        tail_tokens = tokens[-tail_len:]
        tokens = head_tokens + tail_tokens
    else:
        # No truncation needed; keep all tokens
        tokens = tokens[:content_max]

    # Add special tokens
    input_ids = [tokenizer.cls_token_id] + tokens + [tokenizer.sep_token_id]

    # Create attention mask (1 for real tokens, 0 for padding)
    attention_mask = [1] * len(input_ids)

    # Pad to max_len
    padding_len = max_len - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * padding_len
    attention_mask += [0] * padding_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }