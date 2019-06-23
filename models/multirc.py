import collections
import sys
from functools import partial

from modules.bert_module import BertLastCLSModule, BertModule
from task_config import SuperGLUE_LABEL_MAPPING, SuperGLUE_TASK_METRIC_MAPPING
from torch import nn

from snorkel.model.metrics import metric_score
from snorkel.mtl.scorer import Scorer
from snorkel.mtl.task import Task, Operation

from . import utils

sys.path.append("..")  # Adds higher directory to python modules path.


TASK_NAME = "MultiRC"


# custom_metric_funcs #################


def em(golds, probs, preds, uids):
    gt_pds = collections.defaultdict(list)

    raise NotImplementedError("uids were removed; how should we do this now?")
    for gold, pred, uid in zip(golds, preds, uids):
        qid = "%%".join(uid.split("%%")[:2])
        gt_pds[qid].append((gold, pred))

    cnt, tot = 0, 0
    for gt_pd in gt_pds.values():
        tot += 1
        gt, pd = list(zip(*gt_pd))
        if gt == pd:
            cnt += 1

    return cnt / tot


def em_f1(golds, probs, preds):
    f1 = metric_score(golds, probs, preds, metric="f1")
    exact = em(golds, probs, preds)

    return (exact + f1["f1"]) / 2


#########################################


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

    custom_metric_funcs = {"em": em, "em_f1": em_f1}

    loss_fn = partial(utils.ce_loss, f"{TASK_NAME}_pred_head")
    output_fn = partial(utils.output, f"{TASK_NAME}_pred_head")

    task = Task(
        name=TASK_NAME,
        module_pool=nn.ModuleDict(
            {
                "bert_module": bert_module,
                "bert_last_CLS": BertLastCLSModule(
                    dropout_prob=last_hidden_dropout_prob
                ),
                f"{TASK_NAME}_pred_head": nn.Linear(bert_output_dim, task_cardinality),
            }
        ),
        task_flow=[
            Operation(
                name=f"{TASK_NAME}_bert_module",
                module_name="bert_module",
                inputs=[("_input_", "token_ids"), ("_input_", "token_segments"), ("_input_", "token_masks")],
            ),
            Operation(
                name=f"{TASK_NAME}_bert_last_CLS",
                module_name="bert_last_CLS",
                inputs=[(f"{TASK_NAME}_bert_module", 0)],
            ),
            Operation(
                name=f"{TASK_NAME}_pred_head",
                module_name=f"{TASK_NAME}_pred_head",
                inputs=[(f"{TASK_NAME}_bert_last_CLS", 0)],
            ),
        ],
        loss_func=loss_fn,
        output_func=output_fn,
        scorer=Scorer(metrics=metrics, custom_metric_funcs=custom_metric_funcs),
    )

    return task
