import json
import logging
import sys

import numpy as np
import torch
from task_config import SuperGLUE_LABEL_MAPPING

from snorkel.mtl.data import MultitaskDataset

sys.path.append("..")  # Adds higher directory to python modules path.


logger = logging.getLogger(__name__)

TASK_NAME = "WiC"


def get_rows(jsonl_path, max_data_samples):
    logger.info(f"Loading data from {jsonl_path}.")
    rows = [json.loads(row) for row in open(jsonl_path, encoding="utf-8")]
    for i in range(2):
        logger.info(f"Sample {i}: {rows[i]}")

    # Truncate to max_data_samples
    if max_data_samples:
        rows = rows[:max_data_samples]
        logger.info(f"Truncating to {max_data_samples} samples.")

    for row in rows:
        row["sentence1_idx"] = int(row["sentence1_idx"])
        row["sentence2_idx"] = int(row["sentence2_idx"])
        row["label"] = row["label"] if "label" in row else True

    return rows


def parse_from_rows(rows, tokenizer, max_sequence_length):

    # sentence1 text
    sentence1s = []
    # sentence2 text
    sentence2s = []
    # sentence1 idx
    sentence1_idxs = []
    # sentence2 idx
    sentence2_idxs = []
    # word in common
    words = []
    # pos tag
    poses = []
    # label
    labels = []

    token1_idxs = []
    token2_idxs = []

    bert_token_ids = []
    bert_token_masks = []
    bert_token_segments = []

    # Check the maximum token length
    max_len = -1

    for row in rows:
        index = row["idx"]

        sentence1 = row["sentence1"]
        sentence2 = row["sentence2"]
        word = row["word"]
        pos = row["pos"]
        sentence1_idx = row["sentence1_idx"]
        sentence2_idx = row["sentence2_idx"]
        label = row["label"]

        sentence1s.append(sentence1)
        sentence2s.append(sentence2)
        sentence1_idxs.append(sentence1_idx)
        sentence2_idxs.append(sentence2_idx)
        words.append(word)
        poses.append(pos)
        labels.append(SuperGLUE_LABEL_MAPPING[TASK_NAME][label])

        # Tokenize sentences
        sent1_tokens = tokenizer.tokenize(sentence1)
        sent2_tokens = tokenizer.tokenize(sentence2)

        word_tokens_in_sent1 = tokenizer.tokenize(sentence1.split()[sentence1_idx])
        word_tokens_in_sent2 = tokenizer.tokenize(sentence2.split()[sentence2_idx])

        while True:
            total_length = len(sent1_tokens) + len(sent2_tokens)
            if total_length > max_len:
                max_len = total_length
            # Account for [CLS], [SEP], [SEP] with "- 3"
            if total_length <= max_sequence_length - 3:
                break
            if len(sent1_tokens) > len(sent2_tokens):
                sent1_tokens.pop()
            else:
                sent2_tokens.pop()

        for idx in range(sentence1_idx - 1, len(sent1_tokens)):
            if (
                sent1_tokens[idx : idx + len(word_tokens_in_sent1)]
                == word_tokens_in_sent1
            ):
                token1_idxs.append(idx + 1)  # Add [CLS]
                break

        for idx in range(sentence2_idx - 1, len(sent2_tokens)):
            if (
                sent2_tokens[idx : idx + len(word_tokens_in_sent2)]
                == word_tokens_in_sent2
            ):
                token2_idxs.append(
                    idx + len(sent1_tokens) + 2
                )  # Add the length of the first sentence and [CLS] + [SEP]
                break

        # Convert to BERT manner
        tokens = ["[CLS]"] + sent1_tokens + ["[SEP]"]
        token_segments = [0] * len(tokens)

        tokens += sent2_tokens + ["[SEP]"]
        token_segments += [1] * (len(sent2_tokens) + 1)

        token_ids = tokenizer.convert_tokens_to_ids(tokens)

        # Generate mask where 1 for real tokens and 0 for padding tokens
        token_masks = [1] * len(token_ids)

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
            "sentence1": sentence1s,
            "sentence2": sentence2s,
            "word": words,
            "pos": poses,
            "sentence1_idx": sentence1_idxs,
            "sentence2_idx": sentence2_idxs,
            "token1_idx": token1_idxs,
            "token2_idx": token2_idxs,
            "token_ids": bert_token_ids,
            "token_masks": bert_token_masks,
            "token_segments": bert_token_segments,
        },
        Y_dict={"labels": labels},
    )


def parse(jsonl_path, tokenizer, max_data_samples, max_sequence_length):
    rows = get_rows(jsonl_path, max_data_samples)
    return parse_from_rows(rows, tokenizer, max_sequence_length)
