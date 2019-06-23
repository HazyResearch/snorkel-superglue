"""Load data, create a model, (optionally train it), and evaluate it

Example:
```
python run.py --task WiC --n_epochs 1 --counter_unit epochs --evaluation_freq 0.25 --checkpointing 1 --logging 1 --lr 1e-5
```

"""


import argparse
import logging
import os
import sys
from functools import partial

import superglue_tasks
from dataloaders import get_dataloaders
from snorkel.mtl.trainer import Trainer
from snorkel.mtl.model import MultitaskModel
from snorkel.mtl.loggers import TensorBoardWriter
from snorkel.mtl.snorkel_config import default_config
from utils import str2list, str2bool, write_to_file, add_flags_from_config


logging.basicConfig(level=logging.INFO)


def add_application_args(parser):

    parser.add_argument("--task", type=str2list, required=True, help="GLUE tasks")

    parser.add_argument("--log_root", type=str, default="logs", help="Path to root of the logs directory")
    parser.add_argument("--run_name", type=str, help="Name of the current run (can include subdirectories)")
    parser.add_argument(
        "--data_dir", type=str, default="data", help="The path to GLUE dataset"
    )

    parser.add_argument(
        "--bert_model",
        type=str,
        default="bert-large-cased",
        help="Which pretrained BERT model to use",
    )

    parser.add_argument("--batch_size", type=int, default=16, help="batch size")

    parser.add_argument(
        "--slice_hidden_dim", type=int, default=1024, help="Slice hidden dimension size"
    )

    parser.add_argument(
        "--max_data_samples", type=int, default=None, help="Maximum data samples to use"
    )

    parser.add_argument(
        "--max_sequence_length", type=int, default=256, help="Maximum sentence length"
    )

    parser.add_argument(
        "--last_hidden_dropout_prob", type=float, default=0.0, help="Dropout on last layer of bert."
    )

    parser.add_argument(
        "--train", type=str2bool, default=True, help="Whether to train the model"
    )


def get_parser():
    # Parse cmdline args and setup environment
    parser = argparse.ArgumentParser(
        "SuperGLUE Runner", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    add_application_args(parser)
    return parser


def main(args):
    config = vars(args)

    # Full log path gets created in LogWriter
    log_writer = TensorBoardWriter(log_root=args.log_root, run_name=args.run_name)
    config["log_dir"] = log_writer.log_dir
    
    # Save command line argument into file
    cmd_msg = " ".join(sys.argv)
    logging.info(f"COMMAND: {cmd_msg}")
    log_writer.write_text(cmd_msg, "cmd.txt")

    # Save config into file
    logging.info(f"CONFIG: {config}")
    log_writer.write_config(config)

    # Construct dataloaders and tasks and load slices
    dataloaders = []
    tasks = []

    for task_name in args.task:
        task_dataloaders = get_dataloaders(
            data_dir=args.data_dir,
            task_name=task_name,
            splits=["train", "valid", "test"],
            max_sequence_length=args.max_sequence_length,
            max_data_samples=args.max_data_samples,
            tokenizer_name=args.bert_model,
            batch_size=args.batch_size,
        )
        dataloaders.extend(task_dataloaders)

        task = superglue_tasks.task_funcs[task_name](
            args.bert_model, 
            last_hidden_dropout_prob=args.last_hidden_dropout_prob
        )
        tasks.append(task)

    # Build model model
    model = MultitaskModel(name=f"SuperGLUE", tasks=tasks)

    # Load pretrained model if necessary
    if config["model_path"]:
        model.load(config["model_path"])

    # Training
    if args.train:
        trainer = Trainer(**config)
        trainer.train_model(model, dataloaders)

    scores = model.score(dataloaders)

    # Save metrics into file
    logging.info(f"Metrics: {scores}")
    log_writer.write_json(scores, "metrics.txt")

    # Save best metrics into file
    if args.train:
        logging.info(
            f"Best metrics: "
            f"{trainer.log_manager.checkpointer.best_metric_dict}"
        )
        log_writer.write_json(
            trainer.log_manager.checkpointer.best_metric_dict,
            "best_metrics.txt",
        )


if __name__ == "__main__":
    parser = get_parser()
    add_flags_from_config(parser, default_config)
    args = parser.parse_args()
    main(args) 
    
