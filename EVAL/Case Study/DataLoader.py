import pandas as pd
import numpy as np
import torch


def Predictor_dataloader(file):
    data = []
    with open(file, 'r') as f:
        for line in f.readlines():
            data.append(list(map(lambda x: int(x), line.strip(" ").split(" "))))
    data = np.array(data)
    tim = torch.tensor(np.array([list(range(0, 23))] * len(data)))
    inp = torch.tensor(data[:, 1:])
    inp = torch.stack([inp, tim], dim=2)
    target = torch.tensor(data[:, :-1])
    return inp, target
