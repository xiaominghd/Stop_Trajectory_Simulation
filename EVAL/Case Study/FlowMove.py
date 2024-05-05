from collections import Counter
import matplotlib.pyplot as plt

real_file = "/mnt/data/gonghaofeng/deeplearning_project/ODseq_GAN_remote/data/CDR_HaiNan/HaiKou/real.data"
OD_seqGAN = "/mnt/data/gonghaofeng/deeplearning_project/ODseq_GAN_remote/Gen_data/HaiKou/OD_seqGAN/OD_seq.data"
Movsim = "/mnt/data/gonghaofeng/deeplearning_project/ODseq_GAN_remote/Gen_data/HaiKou/Movesim/gene.data"
SeqGAN = "/mnt/data/gonghaofeng/deeplearning_project/ODseq_GAN_remote/Gen_data/HaiKou/SeqGAN/gene.data"
SVAE = "/mnt/data/gonghaofeng/deeplearning_project/ODseq_GAN_remote/Gen_data/HaiKou/SVAE/SVAE_gen.data"
TrajGAIL = "/mnt/data/gonghaofeng/deeplearning_project/ODseq_GAN_remote/Gen_data/HaiKou/TrajGAIL/gen.data"


def f(x):
    col = x // 50
    row = x % 50
    col = col // 2
    row = row // 2
    return col * 25 + row


def view_data(file):
    data = []
    with open(file, 'r') as f1:
        for line in f1.readlines():
            line = line.strip("\n").strip(" ").split(" ")
            line = [f(int(x)) for x in line]
            data.append(line)
    data_len = len(data)

    mid = []
    for i in range(24):
        flow = Counter([d[i] for d in data])
        mid.append(flow[473] / data_len)
    return mid


def ca_distance(s1, s2):

    res = 0
    for i in range(1, len(s1)):
        l1 = (s1[i] - s1[i - 1]) / s1[i - 1]
        l2 = (s2[i] - s2[i - 1]) / s2[i - 1]
        res += abs(l1 - l2)
    return res


r = view_data(real_file)
OD = view_data(OD_seqGAN)
mo = view_data(Movsim)
se = view_data(SeqGAN)
sv = view_data(SVAE)
tr = view_data(TrajGAIL)


plt.plot(r)
plt.show()



#
#
# print(ca_distance(r, OD))
# print(ca_distance(r, mo))
# print(ca_distance(r, se))
# print(ca_distance(r, sv))
# print(ca_distance(r, tr))
