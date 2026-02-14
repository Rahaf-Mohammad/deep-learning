import random
import numpy as np
import torch as tr

seed = 4310386
random.seed(seed)
np.random.seed(seed)
tr.manual_seed(seed)


class SimpleFeedForwardNet(tr.nn.Module):

    def __init__(self):
        super().__init__()

        self.linear1 = tr.nn.Linear(784, 256)   
        self.linear2 = tr.nn.Linear(256, 128)
        self.linear3 = tr.nn.Linear(128, 64)
        self.linear4 = tr.nn.Linear(64, 10)  

    def forward(self, x):

        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        return x


model = SimpleFeedForwardNet()

optimizer = tr.optim.SGD(model.parameters(), lr=0.01)

loss_fn = tr.nn.CrossEntropyLoss()



