import sys
from functools import partial

from modules.bert_module import BertLastCLSModule, BertModule
from modules.copa_module import ChoiceModule
from task_config import SuperGLUE_LABEL_MAPPING, SuperGLUE_TASK_METRIC_MAPPING
from torch import nn

from snorkel.mtl.scorer import Scorer
from snorkel.mtl.task import Task

from . import utils

sys.path.append("..")  # Adds higher directory to python modules path.


TASK_NAME = "SWAG"


def build_model(bert_model_name, last_hidden_dropout_prob=0.0):

    bert_module = BertModule(bert_model_name)
    bert_output_dim = 768 if "base" in bert_model_name else 1024

    task_cardinality = (
        len(SuperGLUE_LABEL_MAPPING[TASK_NAME].keys())
        if SuperGLUE_LABEL_MAPPING[TASK_NAME] is not None
        else 1
    )

    metrics = (
        SuperGLUE_TASK_METRIC_MAPPING[TASK_NAME]
        if TASK_NAME in SuperGLUE_TASK_METRIC_MAPPING
        else []
    )

    custom_metric_funcs = {}

    loss_fn = partial(utils.ce_loss, f"{TASK_NAME}_pred_head")
    output_fn = partial(utils.output, f"{TASK_NAME}_pred_head")

    task = Task(
        name=TASK_NAME,
        module_pool=nn.ModuleDict(
            {
                "bert_module": bert_module,
                "bert_last_cls": BertLastCLSModule(
                    dropout_prob=last_hidden_dropout_prob
                ),
                "linear_module": nn.Linear(bert_output_dim, 1),
                f"{TASK_NAME}_pred_head": ChoiceModule(task_cardinality),
            }
        ),
        task_flow=[
            {
                "name": "choice0",
                "module": "bert_module",
                "inputs": [("_input_", "token1_ids")],
            },
            {
                "name": "choice1",
                "module": "bert_module",
                "inputs": [("_input_", "token2_ids")],
            },
            {
                "name": "choice2",
                "module": "bert_module",
                "inputs": [("_input_", "token3_ids")],
            },
            {
                "name": "choice3",
                "module": "bert_module",
                "inputs": [("_input_", "token4_ids")],
            },
            {
                "name": "choice0_bert_last_cls",
                "module": "bert_last_cls",
                "inputs": [("choice0", 0)],
            },
            {
                "name": "choice1_bert_last_cls",
                "module": "bert_last_cls",
                "inputs": [("choice1", 0)],
            },
            {
                "name": "choice2_bert_last_cls",
                "module": "bert_last_cls",
                "inputs": [("choice2", 0)],
            },
            {
                "name": "choice3_bert_last_cls",
                "module": "bert_last_cls",
                "inputs": [("choice3", 0)],
            },
            {
                "name": "choice0rep",
                "module": "linear_module",
                "inputs": [("choice0_bert_last_cls", 0)],
            },
            {
                "name": "choice1rep",
                "module": "linear_module",
                "inputs": [("choice1_bert_last_cls", 0)],
            },
            {
                "name": "choice2rep",
                "module": "linear_module",
                "inputs": [("choice2_bert_last_cls", 0)],
            },
            {
                "name": "choice3rep",
                "module": "linear_module",
                "inputs": [("choice3_bert_last_cls", 0)],
            },
            {
                "name": f"{TASK_NAME}_pred_head",
                "module": f"{TASK_NAME}_pred_head",
                "inputs": [],
            },
        ],
        loss_func=loss_fn,
        output_func=output_fn,
        scorer=Scorer(metrics=metrics, custom_metric_funcs=custom_metric_funcs),
    )

    return task
