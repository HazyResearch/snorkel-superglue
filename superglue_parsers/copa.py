import json
import logging
import sys

import numpy as np
import torch
from task_config import SuperGLUE_LABEL_MAPPING

from snorkel.mtl.data import MultitaskDataset

sys.path.append("..")  # Adds higher directory to python modules path.


logger = logging.getLogger(__name__)

TASK_NAME = "COPA"


def parse(jsonl_path, tokenizer, max_data_samples, max_sequence_length):
    logger.info(f"Loading data from {jsonl_path}.")
    rows = [json.loads(row) for row in open(jsonl_path, encoding="utf-8")]
    for i in range(2):
        logger.info(f"Sample {i}: {rows[i]}")

    # Truncate to max_data_samples
    if max_data_samples:
        rows = rows[:max_data_samples]
        logger.info(f"Truncating to {max_data_samples} samples.")

    # sentence1
    sent1s = []
    # sentence2
    sent2s = []
    # choice1
    choice1s = []
    # choice2
    choice2s = []

    labels = []

    bert_token1_ids = []
    bert_token2_ids = []

    # Check the maximum token length
    max_len = -1

    for sample in rows:
        index = sample["idx"]
        sent1 = sample["premise"]
        sent2 = sample["question"]

        sent2 = (
            "What was the cause of this?"
            if sent2 == "cause"
            else "What happened as a result?"
        )

        choice1 = sample["choice1"]
        choice2 = sample["choice2"]
        label = sample["label"] if "label" in sample else True
        sent1s.append(sent1)
        sent2s.append(sent2)
        choice1s.append(choice1)
        choice2s.append(choice2)
        labels.append(SuperGLUE_LABEL_MAPPING[TASK_NAME][label])

        # Tokenize sentences
        sent1_tokens = tokenizer.tokenize(sent1)
        sent2_tokens = tokenizer.tokenize(sent2)

        # Tokenize choices
        choice1_tokens = tokenizer.tokenize(choice1)
        choice2_tokens = tokenizer.tokenize(choice2)

        # Convert to BERT manner
        tokens1 = (
            ["[CLS]"]
            + sent1_tokens
            + ["[SEP]"]
            + sent2_tokens
            + ["[SEP]"]
            + choice1_tokens
            + ["[SEP]"]
        )
        tokens2 = (
            ["[CLS]"]
            + sent1_tokens
            + ["[SEP]"]
            + sent2_tokens
            + ["[SEP]"]
            + choice2_tokens
            + ["[SEP]"]
        )

        token1_ids = tokenizer.convert_tokens_to_ids(tokens1)
        token2_ids = tokenizer.convert_tokens_to_ids(tokens2)

        if len(token1_ids) > max_len:
            max_len = len(token1_ids)
        if len(token2_ids) > max_len:
            max_len = len(token2_ids)

        bert_token1_ids.append(torch.LongTensor(token1_ids))
        bert_token2_ids.append(torch.LongTensor(token2_ids))

    labels = torch.from_numpy(np.array(labels))

    logger.info(f"Max token len {max_len}")

    return MultitaskDataset(
        name="SuperGLUE",
        X_dict={
            "sentence1": sent1s,
            "sentence2": sent2s,
            "choice1": choice1s,
            "choice2": choice2s,
            "token1_ids": bert_token1_ids,
            "token2_ids": bert_token2_ids,
        },
        Y_dict={"labels": labels},
    )
