import torch
import torch.nn as nn
from NewModel import ResNet
import torch.optim as optim
import DataLoader
import pandas as pd
import Train


def TrajLen(traject):
    i = 0
    while i < 10 and traject[i][0] != 0:
        i += 1
    return min(i, 10)


class single_dis(nn.Module):

    def __init__(self, point_size=2500, loc_emb_dim=16, tim_emb_dim=16, input_dim=5):
        super(single_dis, self).__init__()
        self.emb_loc = nn.Embedding(point_size, loc_emb_dim)
        self.emb_arr = nn.Embedding(24, tim_emb_dim)
        self.emb_dur = nn.Embedding(24, tim_emb_dim)
        self.fc1 = nn.Sequential(
            ResNet(loc_emb_dim),
            ResNet(loc_emb_dim),
            nn.Linear(loc_emb_dim, input_dim)
        )
        self.fc2 = nn.Sequential(
            ResNet(tim_emb_dim),
            ResNet(tim_emb_dim),
            nn.Linear(tim_emb_dim, input_dim)
        )
        self.fc3 = nn.Sequential(
            ResNet(tim_emb_dim),
            ResNet(tim_emb_dim),
            nn.Linear(tim_emb_dim, input_dim)
        )
        self.fc4 = nn.Sequential(
            ResNet(input_dim * 2),
            ResNet(input_dim * 2),
            nn.Linear(input_dim * 2, 1)
        )

    def forward(self, x):
        loc = self.fc1(self.emb_loc(x[0]))
        arr = self.fc2(self.emb_arr(x[1]))
        # dur = self.fc3(self.emb_dur(x[2]))
        point = torch.concat([loc, arr], dim=0)
        out = torch.sigmoid(self.fc4(point))

        return out

    def pretrain(self, inp, target):
        batch_size = inp.size()[0]
        loss_fn = nn.BCELoss()
        loss = 0
        for i in range(batch_size):
            out = self.forward(inp[i])
            loss += loss_fn(out, target[i])
        return loss / batch_size


class seq_dis(nn.Module):

    def __init__(self, point_size=2500, loc_emb_dim=16, tim_emb_dim=16, input_dim=5, hidden_dim=16, mode="seq"):
        super(seq_dis, self).__init__()
        self.emb_loc = nn.Embedding(point_size, loc_emb_dim)
        self.emb_arr = nn.Embedding(24, tim_emb_dim)
        self.emb_dur = nn.Embedding(24, tim_emb_dim)
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Sequential(
            ResNet(loc_emb_dim),
            ResNet(loc_emb_dim),
            nn.Linear(loc_emb_dim, input_dim)
        )
        self.fc2 = nn.Sequential(
            ResNet(tim_emb_dim),
            ResNet(tim_emb_dim),
            nn.Linear(tim_emb_dim, input_dim)
        )
        self.fc3 = nn.Sequential(
            ResNet(tim_emb_dim),
            ResNet(tim_emb_dim),
            nn.Linear(tim_emb_dim, input_dim)
        )
        if mode == "seq":
            self.fc4 = nn.Linear(input_dim * 3, hidden_dim)
        else:
            self.fc4 = nn.Linear(input_dim, hidden_dim)
        self.GRU = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, bidirectional=True)
        self.GRU2out = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1),
        )
        self.mode = mode

    def forward(self, x):

        x = x[:TrajLen(x)]
        loc = self.fc1(self.emb_loc(x[:, 0]))
        arr = self.fc2(self.emb_arr(x[:, 1]))
        dur = self.fc3(self.emb_dur(x[:, 2]))
        if self.mode == "loc":
            point = loc
        elif self.mode == "arr":
            point = arr
        elif self.mode == "dur":
            point = dur
        else:
            point = torch.concat([loc, arr, dur], dim=1)
        point = self.fc4(point).view(1, -1, self.hidden_dim)
        out, hidden = self.GRU(point)
        s = torch.sigmoid(self.GRU2out(out[:, -1, :].view(2 * self.hidden_dim)))
        return s

    def pretrain(self, inp, target):

        batch_size = inp.size()[0]
        loss_fn = nn.BCELoss()

        loss = 0

        for i in range(batch_size):
            out = self.forward(inp[i])
            loss += loss_fn(out.view(-1), target[i])

        return loss / batch_size

    def reward(self, inp):
        classify = torch.Tensor(inp.size()[0])

        for i in range(len(inp)):
            out = self.forward(inp[i])

            classify[i] = out

        return classify


#
# class class_dis(nn.Module):
#
#     def __init__(self, point_size=2500, loc_emb_dim=16, tim_emb_dim=16, input_dim=5, hidden_dim=16):
#         super(class_dis, self).__init__()
#         self.emb_loc = nn.Embedding(point_size, loc_emb_dim)
#         self.emb_arr = nn.Embedding(24, tim_emb_dim)
#         self.emb_dur = nn.Embedding(24, tim_emb_dim)
#         self.hidden_dim = hidden_dim
#         self.fc1 = nn.Sequential(
#             ResNet(loc_emb_dim),
#             ResNet(loc_emb_dim),
#             nn.Linear(loc_emb_dim, input_dim)
#         )
#         self.fc2 = nn.Sequential(
#             ResNet(tim_emb_dim),
#             ResNet(tim_emb_dim),
#             nn.Linear(tim_emb_dim, input_dim)
#         )
#         self.fc3 = nn.Sequential(
#             ResNet(tim_emb_dim),
#             ResNet(tim_emb_dim),
#             nn.Linear(tim_emb_dim, input_dim)
#         )
#         self.fc4 = nn.Linear(input_dim * 3, hidden_dim)
#         self.GRU = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, bidirectional=True)
#         self.GRU2out = nn.Sequential(
#             nn.Linear(hidden_dim * 2, 1),
#         )
#
#     def forward(self, x):
#         t_len = TrajLen(x)
#         x = x[:t_len]
#         loc = self.fc1(self.emb_loc(x[:, 0]))
#         arr = self.fc2(self.emb_arr(x[:, 1]))
#         dur = self.fc3(self.emb_dur(x[:, 2]))
#         point = self.fc4(torch.concat([loc, arr, dur], dim=1)).view(1, -1, self.hidden_dim)
#         out, hidden = self.GRU(point)
#         s = torch.sigmoid(self.GRU2out(out[:, -1, :].view(2 * self.hidden_dim)))
#
#         return s, out.view(t_len, -1)[:, :self.hidden_dim]
#
#     def pretrain(self, inp, target):
#
#         batch_size = inp.size()[0]
#         loss_fn = nn.BCELoss()
#
#         loss = 0
#
#         for i in range(batch_size):
#             out, _ = self.forward(inp[i])
#             loss += loss_fn(out.view(-1), target[i])
#
#         return loss / batch_size
#
#     def reward(self, inp):
#         classify = torch.Tensor(inp.size()[0])
#
#         for i in range(len(inp)):
#             out, _ = self.forward(inp[i])
#
#             classify[i] = out
#
#         return classify
#
#
# class neighbor_dis(nn.Module):
#
#     def __init__(self, poi_emb=None, poi_emb_dim=16, tim_emb_dim=16, latent_dim=8):
#         super(neighbor_dis, self).__init__()
#         if poi_emb is None:
#             self.poi_emb = nn.Embedding(2500, poi_emb_dim)
#         else:
#             self.poi_emb = poi_emb
#         self.fc1 = nn.Sequential(
#             ResNet(2 * poi_emb_dim),
#             nn.Linear(2 * poi_emb_dim, latent_dim)
#         )
#         self.tim_emb = nn.Embedding(24, tim_emb_dim)
#         self.fc2 = nn.Sequential(
#             ResNet(2 * tim_emb_dim),
#             nn.Linear(2 * tim_emb_dim, latent_dim)
#         )
#         self.fc3 = nn.Sequential(
#             ResNet(2 * latent_dim),
#             nn.Linear(2 * latent_dim, 1)
#         )
#
#     def forward(self, x):
#         poi = self.fc1(self.poi_emb(x[:, 0]).view(-1))
#         tim = self.fc2(self.tim_emb(x[:, 1]).view(-1))
#         p = torch.concat([poi, tim], dim=0)
#         out = torch.sigmoid(self.fc3(p))
#         return out
#
#     def pretrain(self, inp, target):
#         batch_size = inp.size()[0]
#         loss_fn = nn.BCELoss()
#         loss = 0
#         for i in range(batch_size):
#             out = self.forward(inp[i])
#             loss += loss_fn(out.view(-1), target[i])
#         return loss / batch_size
#
#     def reward(self, inp):
#         classify = torch.Tensor(inp.size()[0])
#
#         for i in range(len(inp)):
#             out = self.forward(inp[i])
#
#             classify[i] = out
#
#         return classify
def train(model, inp, target, lr, Epoch, batch_size, file):
    opt = optim.Adam(model.parameters(), lr=lr)

    batch_num = int(inp.size()[0] * 0.8) // batch_size
    for epoch in range(Epoch):

        mid_loss = 0
        for num in range(batch_num):
            opt.zero_grad()
            inp_batch = inp[num * batch_size: num * batch_size + batch_size]
            inp_target = target[num * batch_size: num * batch_size + batch_size]
            loss = model.pretrain(inp_batch, inp_target)
            mid_loss += loss
            loss.backward()
            opt.step()
        print("Epoch:{}/{}   loss={}".format(epoch, Epoch, mid_loss / inp.size()[0]))
        torch.save(model.state_dict(), file)


if __name__ == "__main__":
    r_pos = DataLoader.choice(pd.read_csv("/mnt/data/gonghaofeng/deeplearning_project/ODseq_GAN_remote/data/CDR_HaiNan"
                                          "/HaiKou/real_HaiKou.csv"), num=10000)
    _, _, r_pos_samples = DataLoader.prepare_pretrain(r_pos)
    r_neg = DataLoader.choice(pd.read_csv(r"/mnt/data/gonghaofeng/deeplearning_project/ODseq_GAN_remote/data/CDR_HaiNan"
                                          r"/HaiKou/fake_HaiKou.csv"), num=10000)
    _, _, r_neg_samples = DataLoader.prepare_pretrain(r_neg)
    r_dis_inp, r_dis_target = DataLoader.prepare_dis(r_pos_samples, r_neg_samples)
    model = seq_dis(mode="dur")
    # train(model, r_dis_inp, r_dis_target, lr=1e-3, Epoch=25, batch_size=256,
    #       file=r"/mnt/data/gonghaofeng/deeplearning_project/ODseq_GAN_remote/Result/HaiKou/ODseq_GEN/seq_dis.pth")
    Train.train_dis(model, inp=r_dis_inp, target=r_dis_target, lr=1e-2, Epoch=20, batch_size=256,
                    model_name="/mnt/data/gonghaofeng/deeplearning_project/ODseq_GAN_remote/Result/HaiKou/ODseq_GEN/dur_dis.pth")
