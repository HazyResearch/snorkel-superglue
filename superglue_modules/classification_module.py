from torch import nn


class ClassificationModule(nn.Module):
    def __init__(self, feature_dim, class_cardinality):
        super().__init__()

        self.linear = nn.Linear(feature_dim, class_cardinality)

    def forward(self, feature):
        return self.linear.forward(feature)
