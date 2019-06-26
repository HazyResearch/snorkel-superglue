import json
import logging
import re
import sys

import numpy as np
import torch

from snorkel.mtl.data import MultitaskDataset
from task_config import SuperGLUE_LABEL_MAPPING

sys.path.append("..")  # Adds higher directory to python modules path.


logger = logging.getLogger(__name__)

TASK_NAME = "MultiRC"


def get_rows(jsonl_path, max_data_samples):
    logger.info(f"Loading data from {jsonl_path}.")
    rows = [json.loads(row) for row in open(jsonl_path, encoding="utf-8")]

    new_rows = []

    for row in rows:
        # each example has a paragraph field -> (text, questions)
        # text is the paragraph, which requires some preprocessing
        # questions is a list of questions,
        # has fields (question, sentences_used, answers)
        pid = row["idx"]

        para_sent_list = re.sub(
            "<b>Sent .{1,2}: </b>", "", row["paragraph"]["text"]
        ).split("<br>")

        para = " ".join(para_sent_list)

        for ques in row["paragraph"]["questions"]:
            qid = ques["idx"]
            question = ques["question"]
            sent_used = ques["sentences_used"]

            for ans in ques["answers"]:
                new_row = {}

                aid = ans["idx"]
                answer = ans["text"]

                new_row["pid"] = pid
                new_row["qid"] = qid
                new_row["aid"] = aid
                new_row["paragraph"] = para
                new_row["question"] = question
                new_row["answer"] = answer
                new_row["paragraph_sent_list"] = para_sent_list
                new_row["sent_used"] = sent_used
                new_row["label"] = ans["isAnswer"] if "isAnswer" in ans else False
                new_rows.append(new_row)

    for i in range(2):
        logger.info(f"Sample {i}: {new_rows[i]}")

    # Truncate to max_data_samples
    if max_data_samples:
        new_rows = new_rows[:max_data_samples]
        logger.info(f"Truncating to {max_data_samples} samples.")

    return new_rows


def parse_from_rows(rows, tokenizer, max_sequence_length):

    # paragraph ids
    pids = []
    # question ids
    qids = []
    # answer ids
    aids = []

    # paragraph text
    paras = []
    # question text
    questions = []
    # answer text
    answers = []
    # labels
    labels = []

    bert_token_ids = []
    bert_token_masks = []
    bert_token_segments = []

    # Check the maximum token length
    max_len = -1

    for row in rows:
        pid = row["pid"]
        qid = row["qid"]
        aid = row["aid"]

        para_token = tokenizer.tokenize(row["paragraph"])[: max_sequence_length - 2]
        question_token = tokenizer.tokenize(row["question"])[: max_sequence_length - 2]
        answer_token = tokenizer.tokenize(row["answer"])[: max_sequence_length - 2]

        # Generate tokens
        tokens = (
            ["[CLS]"]
            + para_token
            + ["[SEP]"]
            + question_token
            + answer_token
            + ["[SEP]"]
        )
        # No token segments
        token_segments = [0] * (len(para_token) + 2) + [0] * (
            len(question_token) + len(answer_token) + 1
        )
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        token_masks = [1] * len(token_ids)

        if len(tokens) > max_len:
            max_len = len(tokens)

        # Add to list
        paras.append(row["paragraph"])
        questions.append(row["question"])
        answers.append(row["answer"])

        label = row["label"]
        labels.append(SuperGLUE_LABEL_MAPPING[TASK_NAME][label])

        pids.append(pid)
        qids.append(qid)
        aids.append(aid)

        bert_token_ids.append(torch.LongTensor(token_ids))
        bert_token_masks.append(torch.LongTensor(token_masks))
        bert_token_segments.append(torch.LongTensor(token_segments))

    labels = torch.from_numpy(np.array(labels))

    logger.info(f"Max token len {max_len}")

    return MultitaskDataset(
        name="SuperGLUE",
        X_dict={
            "pid": pids,
            "qid": qids,
            "aid": aids,
            "para": paras,
            "question": questions,
            "answer": answers,
            "token_ids": bert_token_ids,
            "token_masks": bert_token_masks,
            "token_segments": bert_token_segments,
        },
        Y_dict={"labels": labels},
    )


def parse(jsonl_path, tokenizer, max_data_samples, max_sequence_length):
    rows = get_rows(jsonl_path, max_data_samples)
    return parse_from_rows(rows, tokenizer, max_sequence_length)
