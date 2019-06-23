from torch import nn


class RegressionModule(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()

        self.linear = nn.Linear(feature_dim, 1)

    def forward(self, feature):
        return self.linear.forward(feature)
