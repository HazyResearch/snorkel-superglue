import json
import logging
import sys

import numpy as np
import torch
from task_config import SuperGLUE_LABEL_MAPPING

from snorkel.mtl.data import MultitaskDataset

sys.path.append("..")  # Adds higher directory to python modules path.


logger = logging.getLogger(__name__)

TASK_NAME = "WSC"


def get_char_index(text, span_text, span_index):
    tokens = text.replace("\n", " ").lower().split(" ")
    span_tokens = span_text.replace("\n", " ").lower().split(" ")

    # Token exact match
    if tokens[span_index : span_index + len(span_tokens)] == span_tokens:
        st = len(" ".join(tokens[:span_index])) + 1 if span_index != 0 else 0
        ed = st + len(span_text)
        return st, ed

    if span_index < len(tokens):
        # Token fuzzy match with extra chars
        char_in_text = " ".join(tokens[span_index : span_index + len(span_tokens)])
        char_in_span = " ".join(span_tokens)
        if char_in_text.startswith(char_in_span):
            st = len(" ".join(tokens[:span_index])) + 1 if span_index != 0 else 0
            # ed = st + len(char_in_span)
            ed = st + len(char_in_text)
            return st, ed

        # Token fuzzy match with extra chars
        char_in_text = " ".join(tokens[span_index : span_index + len(span_tokens)])
        char_in_span = " ".join(span_tokens)
        if char_in_span.startswith(char_in_text):
            st = len(" ".join(tokens[:span_index])) + 1 if span_index != 0 else 0
            ed = st + len(char_in_text)
            return st, ed

    # Index out of range
    if span_index >= len(tokens):
        span_index -= 10

    # Token fuzzy match with different position
    for idx in range(span_index, len(tokens)):
        if tokens[idx : idx + len(span_tokens)] == span_tokens:
            st = len(" ".join(tokens[:idx])) + 1 if idx != 0 else 0
            ed = st + len(span_text)
            return st, ed

    # Token best fuzzy match with different position
    for idx in range(span_index, len(tokens)):
        if tokens[idx] == span_tokens[0]:
            for length in range(1, len(span_tokens)):
                if tokens[idx : idx + length] != span_tokens[:length]:
                    st = len(" ".join(tokens[:idx])) + 1 if idx != 0 else 0
                    ed = st + len(" ".join(span_tokens[: length - 1]))
                    return st, ed

    return None


def parse(jsonl_path, tokenizer, max_data_samples, max_sequence_length):
    logger.info(f"Loading data from {jsonl_path}.")
    rows = [json.loads(row) for row in open(jsonl_path, encoding="utf-8")]
    for i in range(2):
        logger.info(f"Sample {i}: {rows[i]}")

    # Truncate to max_data_samples
    if max_data_samples:
        rows = rows[:max_data_samples]
        logger.info(f"Truncating to {max_data_samples} samples.")

    # sentence text
    sentences = []
    # span1
    span1s = []
    # span2
    span2s = []
    # span1 idx
    span1_idxs = []
    # span2 idx
    span2_idxs = []
    # label
    labels = []

    token1_idxs = []
    token2_idxs = []

    bert_tokens = []
    bert_token_ids = []
    bert_token_masks = []
    bert_token_segments = []

    # Check the maximum token length
    max_len = -1

    for row in rows:
        index = row["idx"]

        text = row["text"]
        span1_text = row["target"]["span1_text"]
        span2_text = row["target"]["span2_text"]
        span1_index = row["target"]["span1_index"]
        span2_index = row["target"]["span2_index"]

        label = row["label"] if "label" in row else True

        span1_char_index = get_char_index(text, span1_text, span1_index)
        span2_char_index = get_char_index(text, span2_text, span2_index)

        assert span1_char_index is not None, f"Check example {id} in {jsonl_path}"
        assert span2_char_index is not None, f"Check example {id} in {jsonl_path}"

        # Tokenize sentences
        bert_tokens_sub1 = tokenizer.tokenize(
            text[: min(span1_char_index[0], span2_char_index[0])]
        )

        if span1_char_index[0] < span2_char_index[0]:
            bert_tokens_sub2 = tokenizer.tokenize(
                text[span1_char_index[0] : span1_char_index[1]]
            )
            token1_idx = [
                len(bert_tokens_sub1) + 1,
                len(bert_tokens_sub1) + len(bert_tokens_sub2),
            ]
        else:
            bert_tokens_sub2 = tokenizer.tokenize(
                text[span2_char_index[0] : span2_char_index[1]]
            )
            token2_idx = [
                len(bert_tokens_sub1) + 1,
                len(bert_tokens_sub1) + len(bert_tokens_sub2),
            ]

        sub3_st = (
            span1_char_index[1]
            if span1_char_index[0] < span2_char_index[0]
            else span2_char_index[1]
        )
        sub3_ed = (
            span1_char_index[0]
            if span1_char_index[0] > span2_char_index[0]
            else span2_char_index[0]
        )

        bert_tokens_sub3 = tokenizer.tokenize(text[sub3_st:sub3_ed])
        if span1_char_index[0] < span2_char_index[0]:
            bert_tokens_sub4 = tokenizer.tokenize(
                text[span2_char_index[0] : span2_char_index[1]]
            )
            cur_len = (
                len(bert_tokens_sub1) + len(bert_tokens_sub2) + len(bert_tokens_sub3)
            )
            token2_idx = [cur_len + 1, cur_len + len(bert_tokens_sub4)]
        else:
            bert_tokens_sub4 = tokenizer.tokenize(
                text[span1_char_index[0] : span1_char_index[1]]
            )
            cur_len = (
                len(bert_tokens_sub1) + len(bert_tokens_sub2) + len(bert_tokens_sub3)
            )
            token1_idx = [cur_len + 1, cur_len + len(bert_tokens_sub4)]

        if span1_char_index[0] < span2_char_index[0]:
            bert_tokens_sub5 = tokenizer.tokenize(text[span2_char_index[1] :])
        else:
            bert_tokens_sub5 = tokenizer.tokenize(text[span1_char_index[1] :])

        tokens = (
            ["[CLS]"]
            + bert_tokens_sub1
            + bert_tokens_sub2
            + bert_tokens_sub3
            + bert_tokens_sub4
            + bert_tokens_sub5
            + ["[SEP]"]
        )

        if len(tokens) > max_len:
            max_len = len(tokens)

        token_ids = tokenizer.convert_tokens_to_ids(tokens)

        token_segments = [0] * len(token_ids)
        # Generate mask where 1 for real tokens and 0 for padding tokens
        token_masks = [1] * len(token_ids)

        token1_idxs.append(token1_idx)
        token2_idxs.append(token2_idx)

        sentences.append(text)
        span1s.append(span1_text)
        span2s.append(span2_text)
        span1_idxs.append(span1_index)
        span2_idxs.append(span2_index)
        labels.append(SuperGLUE_LABEL_MAPPING[TASK_NAME][label])

        bert_tokens.append(tokens)
        bert_token_ids.append(torch.LongTensor(token_ids))
        bert_token_masks.append(torch.LongTensor(token_masks))
        bert_token_segments.append(torch.LongTensor(token_segments))

    token1_idxs = torch.from_numpy(np.array(token1_idxs))
    token2_idxs = torch.from_numpy(np.array(token2_idxs))

    labels = torch.from_numpy(np.array(labels))

    logger.info(f"Max token len {max_len}")

    return MultitaskDataset(
        name="SuperGLUE",
        X_dict={
            "sentence": sentences,
            "span1": span1s,
            "span2": span2s,
            "span1_idx": span1_idxs,
            "span2_idx": span2_idxs,
            "token1_idx": token1_idxs,
            "token2_idx": token2_idxs,
            "tokens": bert_tokens,
            "token_ids": bert_token_ids,
            "token_masks": bert_token_masks,
            "token_segments": bert_token_segments,
        },
        Y_dict={"labels": labels},
    )
