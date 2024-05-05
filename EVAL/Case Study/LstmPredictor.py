import torch
import torch.nn as nn
import torch.functional as F
import DataLoader
import torch.optim as optim
import sys


class Logger(object):
    def __init__(self, filename='wes.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


class t_Lstm(nn.Module):

    def __init__(self, loc_dim, loc_emb, tim_dim, tim_emb, drop_out, hidden, num_layers):
        super(t_Lstm, self).__init__()
        self.loc_dim = loc_dim
        self.loc_emb = loc_emb
        self.tim_dim = tim_dim
        self.tim_emb = tim_emb
        self.drop_out = drop_out
        self.hidden = hidden
        self.num_layers = num_layers
        self.emb_loc = nn.Embedding(self.loc_dim, self.loc_emb)
        self.emb_tim = nn.Embedding(self.tim_dim, self.tim_emb)
        self.fc1 = nn.Sequential(
            nn.Linear(self.loc_emb + self.tim_emb, self.loc_emb + self.tim_emb),
            nn.Tanh(),
            nn.Linear(self.loc_emb + self.tim_emb, self.loc_emb + self.tim_emb),
            nn.Tanh(),
        )
        self.GRU = nn.GRU(input_size=self.loc_emb + self.tim_emb, hidden_size=self.hidden,
                          num_layers=self.num_layers, dropout=drop_out, batch_first=True)
        self.LSTM2out = nn.Sequential(
            nn.Linear(self.hidden, self.hidden),
            nn.Tanh(),
            nn.Linear(self.hidden, self.loc_dim)
        )

    def init_hidden(self, batch_size=1):
        hidden = torch.autograd.Variable(torch.zeros(self.num_layers, batch_size, self.hidden))
        return hidden

    def forward(self, x, hidden):
        x = torch.permute(x, [1, 0])
        loc = self.emb_loc(x[0])
        tim = self.emb_tim(x[1])
        x = torch.concat([loc, tim], dim=1)
        x = self.fc1(x).view(x.size()[0], 1, -1)
        out, hidden = self.GRU(x, hidden)
        out = self.LSTM2out(out).view(-1, self.loc_dim)
        out = torch.log_softmax(out, dim=1)
        return out, hidden


def train(model, epoch, inp, target, batch_size, val_inp, val_target, optimizer):
    num, seq_len, _ = inp.size()
    batch_num = num // batch_size
    loss_fn = nn.NLLLoss()
    for e in range(epoch):
        mid_loss = 0
        for num in range(batch_num):
            optimizer.zero_grad()
            model.train()  # 申明模型是在训练还是在测试，这里主要是处理一些dropout
            batch_inp = torch.autograd.Variable(inp[num * batch_size: num * batch_size + batch_size])
            batch_target = torch.autograd.Variable(target[num * batch_size:num * batch_size + batch_size])
            hidden = model.init_hidden(batch_size=batch_size)
            loss_t = 0
            for i in range(seq_len):
                out, hidden = model(batch_inp[:, i, :], hidden)
                loss_t += loss_fn(out, batch_target[:, i])
            mid_loss += loss_t
            loss_t.backward()
            optimizer.step()
        loss1, acc1 = test(model, val_inp, val_target, N=5)
        loss2, acc2 = test(model, val_inp, val_target, N=10)
        loss3, acc3 = test(model, val_inp, val_target, N=20)
        print("Train Epoch:{}/{} \t Train Loss: {}\t Validation Loss: "
              "{:.6f} \t validation_acc@{}: {:.6f}\t@{}: {:.6f}\t@{}: {:.6f}".format(e, epoch, mid_loss / num,
                                                                                     loss1, 5, acc1, 10, acc2, 20,
                                                                                     acc3))


def test(model, inp, target, N):
    model.eval()
    num, seq_len, _ = inp.size()
    hidden = model.init_hidden(num)
    loss = 0
    loss_fn = nn.NLLLoss()
    res = 0
    for i in range(seq_len):
        out, hidden = model(inp[:, i, :], hidden)
        loss += loss_fn(out, target[:, i])
        values, indices = torch.topk(torch.exp(out), k=N, dim=1)
        for j in range(len(indices)):
            if target[:, i][j] in indices[j]:
                res += 1
    return loss / num, res / (num * seq_len)


sys.stdout = Logger(stream=sys.stdout)
model = t_Lstm(loc_dim=2500, loc_emb=32, tim_dim=24, tim_emb=32, drop_out=0.6, hidden=32, num_layers=3)
inp, target = DataLoader.Predictor_dataloader(file="/mnt/data/gonghaofeng/deeplearning_project/ODseq_GAN_remote"
                                                   "/Gen_data/HaiKou/Movesim/gene.data")
n = int(len(inp) * 0.8)
opt = optim.Adam(model.parameters(), lr=1e-4)

val_inp, val_target = DataLoader.Predictor_dataloader(file="/mnt/data/gonghaofeng/deeplearning_project/"
                                                           "ODseq_GAN_remote/data/CDR_HaiNan/HaiKou/real.data")

train(model=model, epoch=20, inp=inp, target=target, batch_size=256,
      val_inp=val_inp[:2000], val_target=val_target[:2000], optimizer=opt)
