import numpy as np
from DataLoader.LoadData import prepare_pretrain, prepare_discriminator
from model.GAIL import GAIL
import torch

real = np.load("data/SanYa/train_data.pkl", allow_pickle=True)
real_inp, real_target = prepare_pretrain(real)
gen = GAIL(state_dim=2500, action_dim=2500)
gen.load_state_dict(torch.load("Result/SanYa/GAIL_GAN.pth"))
fake = gen.sample(real_inp[:, 0].view(-1, 1), 24)
FILE = 'gen_SanYa.data'
with open(FILE, 'w') as f:
    for t in fake:
        f.write(' '.join([str(x) for x in t]) + '\n')
