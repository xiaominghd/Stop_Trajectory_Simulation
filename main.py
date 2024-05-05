import numpy as np
import pickle

real = "/mnt/data/gonghaofeng/deeplearning_project/ODseq_GAN_remote/data/CDR_HaiNan/Sanya/real_SanYa_seq.txt"
with open(real, 'r') as f:
    data = []
    for line in f.readlines():
        line = line.strip("\n").strip("  ").split(" ")
        line = list(map(lambda x: int(x), line))
        data.append(line)

with open('real_data.pkl', 'wb') as f:
    pickle.dump(data, f)
