import argparse
import logging
import os

import pandas as pd
from snorkel.mtl.data import MultitaskDataset


def str2list(v, dim=","):
    return [t.strip() for t in v.split(dim)]


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise Exception("Boolean value expected.")


def write_to_file(path, file_name, value):
    if not isinstance(value, str):
        value = str(value)
    fout = open(os.path.join(path, file_name), "w")
    fout.write(value + "\n")
    fout.close()


def add_flags_from_config(parser, config_dict):
    """
    Adds a flag (and default value) to an ArgumentParser for each parameter in a config
    """

    def OrNone(default):
        def func(x):
            # Convert "none" to proper None object
            if x.lower() == "none":
                return None
            # If default is None (and x is not None), return x without conversion as str
            elif default is None:
                return str(x)
            # Otherwise, default has non-None type; convert x to that type
            else:
                return type(default)(x)

        return func

    for param in config_dict:
        default = config_dict[param]
        try:
            if isinstance(default, dict):
                parser = add_flags_from_config(parser, default)
            elif isinstance(default, bool):
                parser.add_argument(f"--{param}", type=str2bool, default=default)
            elif isinstance(default, list):
                if len(default) > 0:
                    # pass a list as argument
                    parser.add_argument(
                        f"--{param}",
                        action="append",
                        type=type(default[0]),
                        default=default,
                    )
                else:
                    parser.add_argument(f"--{param}", action="append", default=default)
            else:
                parser.add_argument(f"--{param}", type=OrNone(default), default=default)
        except argparse.ArgumentError:
            logging.warning(
                f"Could not add flag for param {param} because it was already present."
            )
    return parser


def task_dataset_to_dataframe(dataset: MultitaskDataset) -> pd.DataFrame:
    data_dict = dataset.X_dict
    data_dict["labels"] = dataset.Y_dict["labels"]
    return pd.DataFrame(data_dict)
