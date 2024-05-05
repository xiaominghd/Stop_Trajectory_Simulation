import math
from collections import Counter


def compute_BLEU(candidate, reference):
    reference = [ref.split(" ") for ref in reference]
    for n in range(1, 5):  # 计算的是BLEU1,...,BLEU4
        helper = 0
        for i in range(len(candidate)):
            can = candidate[i].split(" ")
            if i % 100 ==0:
                print(i)
            for j in range(len(can) - n + 1):
                ngram = ' '.join(can[j: j + n])
                helper += max([ref.count(ngram) for ref in reference])
        print(helper / (len(candidate) * (len(candidate[0]) - n + 1)))


with open("/mnt/data/gonghaofeng/deeplearning_project/ODseq_GAN_remote/Gen_data/HaiKou/Movesim/gene.data", 'r') as f:
    candidate = [line.strip("\n") for line in f.readlines()]
with open("/mnt/data/gonghaofeng/deeplearning_project/ODseq_GAN_remote/model/Movsim/data/HaiNan/real.data") as f:
    ref = [line.strip("\n") for line in f.readlines()][:4000]
print(compute_BLEU(candidate, ref))
