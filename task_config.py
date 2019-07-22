SuperGLUE_TASK_NAMES = ["CB", "COPA", "MultiRC", "RTE", "WiC", "WSC"]

SuperGLUE_TASK_SPLIT_MAPPING = {
    "CB": {"train": "train.jsonl", "valid": "val.jsonl", "test": "test.jsonl"},
    "COPA": {"train": "train.jsonl", "valid": "val.jsonl", "test": "test.jsonl"},
    "MultiRC": {"train": "train.jsonl", "valid": "val.jsonl", "test": "test.jsonl"},
    "RTE": {"train": "train.jsonl", "valid": "val.jsonl", "test": "test.jsonl"},
    "WiC": {"train": "train.jsonl", "valid": "val.jsonl", "test": "test.jsonl"},
    "WSC": {"train": "train.jsonl", "valid": "val.jsonl", "test": "test.jsonl"},
    "SWAG": {"train": "train.csv", "valid": "val.csv", "test": "test.csv"},
}

SuperGLUE_LABEL_MAPPING = {
    "CB": {"entailment": 1, "contradiction": 2, "neutral": 3},
    "COPA": {0: 1, 1: 2},
    "RTE": {"entailment": 1, "not_entailment": 2},
    "WiC": {True: 1, False: 2},
    "WSC": {True: 1, False: 2},
    "MultiRC": {True: 1, False: 2},
    "SWAG": {0: 1, 1: 2, 2: 3, 3: 4},
}

SuperGLUE_LABEL_INVERSE = {}
for task, mapping in SuperGLUE_LABEL_MAPPING.items():
    SuperGLUE_LABEL_INVERSE[task] = {v: k for k, v in mapping.items()}

SuperGLUE_TASK_METRIC_MAPPING = {
    "CB": ["accuracy"],
    "COPA": ["accuracy"],
    "MultiRC": ["f1"],
    "RTE": ["accuracy"],
    "WiC": ["accuracy"],
    "WSC": ["accuracy"],
    "SWAG": ["accuracy"],
}
