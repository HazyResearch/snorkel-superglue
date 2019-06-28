import sys
from functools import partial

from superglue_modules.bert_module import BertLastCLSModule, BertModule
from superglue_modules.copa_module import ChoiceModule
from task_config import SuperGLUE_LABEL_MAPPING, SuperGLUE_TASK_METRIC_MAPPING
from torch import nn

from snorkel.mtl.scorer import Scorer
from snorkel.mtl.task import Task, Operation

from . import utils

sys.path.append("..")  # Adds higher directory to python modules path.


TASK_NAME = "COPA"


def build_task(bert_model_name, last_hidden_dropout_prob=0.0):

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
                f"{TASK_NAME}_feature": BertLastCLSModule(
                    dropout_prob=last_hidden_dropout_prob
                ),
                "linear_module": nn.Linear(bert_output_dim, 1),
                f"{TASK_NAME}_pred_head": ChoiceModule(task_cardinality),
            }
        ),
        task_flow=[
            Operation(
                name="choice0",
                module_name="bert_module",
                inputs=[("_input_", "token1_ids")],
            ),
            Operation(
                name="choice1",
                module_name="bert_module",
                inputs=[("_input_", "token2_ids")],
            ),
            Operation(
                name="choice0_bert_last_cls",
                module_name=f"{TASK_NAME}_feature",
                inputs=[("choice0", 0)],
            ),
            Operation(
                name="choice1_bert_last_cls",
                module_name=f"{TASK_NAME}_feature",
                inputs=[("choice1", 0)],
            ),
            Operation(
                name="choice0rep",
                module_name="linear_module",
                inputs=[("choice0_bert_last_cls", 0)],
            ),
            Operation(
                name="choice1rep",
                module_name="linear_module",
                inputs=[("choice1_bert_last_cls", 0)],
            ),
            Operation(
                name=f"{TASK_NAME}_pred_head",
                module_name=f"{TASK_NAME}_pred_head",
                inputs=[],
            ),
        ],
        loss_func=loss_fn,
        output_func=output_fn,
        scorer=Scorer(metrics=metrics, custom_metric_funcs=custom_metric_funcs),
    )

    return task
