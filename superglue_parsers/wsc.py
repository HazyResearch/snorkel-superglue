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

    bert_token_ids = []
    bert_token_masks = []
    bert_token_segments = []

    # Check the maximum token length
    max_len = -1

    for row in rows:
        index = row["idx"]

        sentence = row["text"]
        span1 = row["target"]["span1_text"].strip()
        span2 = row["target"]["span2_text"].strip()
        span1_idx = row["target"]["span1"]
        span2_idx = row["target"]["span2"]

        label = row["label"] if "label" in row else True

        sentences.append(sentence)
        span1s.append(span1)
        span2s.append(span2)
        labels.append(SuperGLUE_LABEL_MAPPING[TASK_NAME][label])

        # Tokenize sentences
        sent_tokens = sentence.split()
        span1_idxs.append([span1_idx[0]+1, span1_idx[1]])
        span2_idxs.append([span2_idx[0]+1, span2_idx[1]])

        # Convert to BERT manner
        tokens = ["[CLS]"] + sent_tokens + ["[SEP]"]

        if len(tokens) > max_len:
            max_len = len(tokens)
        
        token_ids = tokenizer.convert_tokens_to_ids(tokens)

        token_segments = [0] * len(token_ids)
        # Generate mask where 1 for real tokens and 0 for padding tokens
        token_masks = [1] * len(token_ids)

        bert_token_ids.append(torch.LongTensor(token_ids))
        bert_token_masks.append(torch.LongTensor(token_masks))
        bert_token_segments.append(torch.LongTensor(token_segments))

    span1_idxs = torch.from_numpy(np.array(span1_idxs))
    span2_idxs = torch.from_numpy(np.array(span2_idxs))

    labels = torch.from_numpy(np.array(labels))

    logger.info(f"Max token len {max_len}")

    return MultitaskDataset(
        name="SuperGLUE",
        X_dict={
            "sentence": sentences,
            "span1": span1s,
            "span2": span2s,
            "span1_idxs": span1_idxs,
            "span2_idxs": span2_idxs,
            "token_ids": bert_token_ids,
            "token_masks": bert_token_masks,
            "token_segments": bert_token_segments,
        },
        Y_dict={"labels": labels},
    )