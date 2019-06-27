import logging
import os

import superglue_parsers
from task_config import SuperGLUE_TASK_SPLIT_MAPPING
from tokenizer import get_tokenizer

from pytorch_pretrained_bert import BertTokenizer
from snorkel.mtl.data import MultitaskDataLoader


logger = logging.getLogger(__name__)


def get_jsonl_path(data_dir: str, task_name: str, split: str):
    return os.path.join(
        data_dir, task_name, SuperGLUE_TASK_SPLIT_MAPPING[task_name][split]
    )


def get_dataset(
    data_dir: str,
    task_name: str,
    split: str,
    tokenizer: BertTokenizer,
    max_data_samples: int,
    max_sequence_length: int,
):
    jsonl_path = get_jsonl_path(data_dir, task_name, split)
    return superglue_parsers.parser[task_name](
        jsonl_path, tokenizer, max_data_samples, max_sequence_length
    )


def get_dataloaders(
    data_dir,
    task_name="MultiRC",
    splits=["train", "valid", "test"],
    max_data_samples=None,
    max_sequence_length=256,
    tokenizer_name="bert-base-uncased",
    batch_size=16,
):
    """Load data and return dataloaders"""

    dataloaders = []

    tokenizer = get_tokenizer(tokenizer_name)

    for split in splits:
        dataset = get_dataset(
            data_dir, task_name, split, tokenizer, max_data_samples, max_sequence_length
        )
        dataloader = MultitaskDataLoader(
            task_to_label_dict={task_name: "labels"},
            dataset=dataset,
            split=split,
            batch_size=batch_size,
            shuffle=(split == "train"),
        )
        dataloaders.append(dataloader)

        logger.info(f"Loaded {split} for {task_name} with {len(dataset)} samples.")

    return dataloaders
