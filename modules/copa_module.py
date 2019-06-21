import torch
from torch import nn


class ChoiceModule(nn.Module):
    def __init__(self, n_choices=2):
        super().__init__()

        self.n_choices = n_choices

    def forward(self, immediate_ouput_dict):
        logits = []
        for i in range(self.n_choices):
            logits.append(immediate_ouput_dict[f"choice{str(i)}rep"][0])

        logits = torch.cat(logits, dim=1)

        return logits
