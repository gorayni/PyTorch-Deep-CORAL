from __future__ import print_function, division
import torch.nn as nn


class FrozenCNN(nn.Module):
    def __init__(self, features_size=2048, num_classes=17):
        super(FrozenCNN, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(features_size, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
