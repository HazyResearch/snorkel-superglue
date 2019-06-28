import json
import logging
import sys
import pandas as pd

import numpy as np
import torch
from task_config import SuperGLUE_LABEL_MAPPING

from snorkel.mtl.data import MultitaskDataset

sys.path.append("..")  # Adds higher directory to python modules path.


logger = logging.getLogger(__name__)

TASK_NAME = "SWAG"


def parse(csv_path, tokenizer, max_data_samples, max_sequence_length):
    logger.info(f"Loading data from {csv_path}.")
    rows = pd.read_csv(csv_path)

    # for i in range(2):
    #     logger.info(f"Sample {i}: {rows[i]}")

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
    # choice3
    choice3s = []
    # choice4
    choice4s = []

    labels = []

    bert_token1_ids = []
    bert_token2_ids = []
    bert_token3_ids = []
    bert_token4_ids = []

    # Check the maximum token length
    max_len = -1

    for ex_idx, ex in rows.iterrows():
        sent1 = ex["sent1"]
        sent2 = ex["sent2"]

        choice1 = ex["ending0"]
        choice2 = ex["ending1"]
        choice3 = ex["ending2"]
        choice4 = ex["ending3"]

        label = ex["label"] if "label" in ex else 0

        sent1s.append(sent1)
        sent2s.append(sent2)
        choice1s.append(choice1)
        choice2s.append(choice2)
        choice3s.append(choice3)
        choice4s.append(choice4)

        labels.append(SuperGLUE_LABEL_MAPPING[TASK_NAME][label])

        # Tokenize sentences
        sent1_tokens = tokenizer.tokenize(sent1)
        sent2_tokens = tokenizer.tokenize(sent2)
        choice1_tokens = tokenizer.tokenize(choice1)
        choice2_tokens = tokenizer.tokenize(choice2)
        choice3_tokens = tokenizer.tokenize(choice3)
        choice4_tokens = tokenizer.tokenize(choice4)

        # Convert to BERT manner
        bert_token1 = (
            ["[CLS]"]
            + sent1_tokens
            + ["[SEP]"]
            + sent2_tokens
            + choice1_tokens
            + ["[SEP]"]
        )
        bert_token2 = (
            ["[CLS]"]
            + sent1_tokens
            + ["[SEP]"]
            + sent2_tokens
            + choice2_tokens
            + ["[SEP]"]
        )
        bert_token3 = (
            ["[CLS]"]
            + sent1_tokens
            + ["[SEP]"]
            + sent2_tokens
            + choice3_tokens
            + ["[SEP]"]
        )
        bert_token4 = (
            ["[CLS]"]
            + sent1_tokens
            + ["[SEP]"]
            + sent2_tokens
            + choice4_tokens
            + ["[SEP]"]
        )

        token1_ids = tokenizer.convert_tokens_to_ids(bert_token1)
        token2_ids = tokenizer.convert_tokens_to_ids(bert_token2)
        token3_ids = tokenizer.convert_tokens_to_ids(bert_token3)
        token4_ids = tokenizer.convert_tokens_to_ids(bert_token4)

        if len(token1_ids) > max_len:
            max_len = len(token1_ids)
        if len(token2_ids) > max_len:
            max_len = len(token2_ids)
        if len(token3_ids) > max_len:
            max_len = len(token3_ids)
        if len(token4_ids) > max_len:
            max_len = len(token4_ids)

        bert_token1_ids.append(torch.LongTensor(token1_ids))
        bert_token2_ids.append(torch.LongTensor(token2_ids))
        bert_token3_ids.append(torch.LongTensor(token3_ids))
        bert_token4_ids.append(torch.LongTensor(token4_ids))

    labels = torch.from_numpy(np.array(labels))

    logger.info(f"Max token len {max_len}")

    return MultitaskDataset(
        name="SuperGLUE",
        X_dict={
            "sent1": sent1s,
            "sent2": sent2s,
            "choice1": choice1s,
            "choice2": choice2s,
            "choice3": choice3s,
            "choice4": choice4s,
            "token1_ids": bert_token1_ids,
            "token2_ids": bert_token2_ids,
            "token3_ids": bert_token3_ids,
            "token4_ids": bert_token4_ids,
        },
        Y_dict={"labels": labels},
    )
