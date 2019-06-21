from . import cb, copa, multirc, rte, wic, wsc, swag, semcor

model = {
    "CB": cb.build_model,
    "COPA": copa.build_model,
    "MultiRC": multirc.build_model,
    "RTE": rte.build_model,
    "WiC": wic.build_model,
    "WSC": wsc.build_model,
    "SWAG": swag.build_model,
    "SemCor": semcor.build_model,
}
