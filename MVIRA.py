import torch
import copy
import Config
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import utils as nn_utils
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pickle
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import math
from torch.nn.utils import weight_norm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# 我们神经网络的参数一开始都是随机初始化的，然后才开始训练，如果不设置随机数种子，那么每一次训练的最终结果都会有差别
# 设置随机数种子以后，相当于无论跑多少次，随机的结果都是一样，最后训练的也是一样
# 目的是为了这套代码在谁的电脑上跑，最后结果都差不多，用来证实你的结果和模型匹配
RANDOM_SEED = Config.RANDOM_SEED  # 设置随机数种子
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
batch_size = Config.batch_size
# batch_size = 512
input_size = Config.input_size
embed_dim = Config.embed_dim
# embed_dim = 2 * int(np.floor(np.power(input_size, 0.25)))
num_layers = Config.num_layers
hidden_size = Config.hidden_size
workers = Config.workers
learning_rate = Config.learning_rate
# learning_rate = 1e-3 #学习率，越高梯度下降的越快
epochs = Config.epochs  # 总共训练多少遍
device = Config.device
datatype = Config.datatype

#save_model_dir = '/home/lb/Model/mimic_Uncertainlymodel_5.0_' + datatype + '.pth'  # 模型保存的路径
save_model_dir = '/home/lb/Model/mimic_Uncertainlymodel_6.0_' + datatype + '.pth'  # 模型保存的路径

alpha_loss1 = 1
alpha_loss2 = 10

alpha_loss3 = 0.0001
testepoch = 5
if datatype == "mimic3":
    batch_size = 256
    learning_rate = 1e-2
else:
    batch_size = 512
    learning_rate = 1e-2
    testepoch = 5



'''输入：
input: [seq_len, batch, input_size] 如果batch_first=true，那么batch在第一个
h0 : [num_layers* num_directions, batch, hidden_size]
输出：
output: [seq_len, batch, num_directions * hidden_size] 如果batch_first=true，那么batch在第一个
hn: [num_layers * num_directions, batch, hidden_size]'''


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


def get_data():
    # print(np.arange(100))
    x_torch = pickle.load(open('dataset/lb_' + datatype + '_x_for_missingvalue.p', 'rb'))
    m_torch = pickle.load(open('dataset/lb_' + datatype + '_m_for_missingvalue.p', 'rb'))
    delta_torch = pickle.load(open('dataset/lb_' + datatype + '_delta_for_missingvalue.p', 'rb'))
    y_torch = pickle.load(open('dataset/lb_' + datatype + '_y.p', 'rb'))
    x_lens = pickle.load(open('dataset/lb_' + datatype + '_len.p', 'rb'))
    print(x_torch.shape)
    print(m_torch.shape)
    print(delta_torch.shape)
    print(y_torch.shape)
    print(x_lens.shape)
    # 划分训练集验证集和测试集，8：1：1
    train_ratio = 0.8
    valid_ratio = 0.1
    test_ratio = 0.1
    N = len(x_torch)

    training_x = x_torch[: int(train_ratio * N)]
    validing_x = x_torch[int(train_ratio * N): int(
        (train_ratio + valid_ratio) * N)]
    testing_x = x_torch[int((train_ratio + valid_ratio) * N):]

    training_m = m_torch[: int(train_ratio * N)]
    validing_m = m_torch[int(train_ratio * N): int(
        (train_ratio + valid_ratio) * N)]
    testing_m = m_torch[int((train_ratio + valid_ratio) * N):]

    training_delta = delta_torch[: int(train_ratio * N)]
    validing_delta = delta_torch[int(train_ratio * N): int(
        (train_ratio + valid_ratio) * N)]
    testing_delta = delta_torch[int((train_ratio + valid_ratio) * N):]

    training_x_lens = x_lens[: int(train_ratio * N)]
    validing_x_lens = x_lens[int(train_ratio * N): int(
        (train_ratio + valid_ratio) * N)]
    testing_x_lens = x_lens[int((train_ratio + valid_ratio) * N):]

    training_y = y_torch[: int(train_ratio * N)]
    validing_y = y_torch[int(train_ratio * N): int(
        (train_ratio + valid_ratio) * N)]
    testing_y = y_torch[int((train_ratio + valid_ratio) * N):]

    train_deal_dataset = TensorDataset(training_x, training_y, training_m, training_delta, training_x_lens)

    train_loader = DataLoader(dataset=train_deal_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True,
                              num_workers=workers)

    test_deal_dataset = TensorDataset(testing_x, testing_y, testing_m, testing_delta, testing_x_lens)

    test_loader = DataLoader(dataset=test_deal_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             drop_last=True,
                             num_workers=workers)

    valid_deal_dataset = TensorDataset(validing_x, validing_y, validing_m, validing_delta, validing_x_lens)

    valid_loader = DataLoader(dataset=valid_deal_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True,
                              num_workers=workers)

    return train_loader, test_loader, valid_loader

class VRAE(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super(VRAE, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        #encoder
        self.w_enc = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.w_in = torch.nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.w_miu = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.w_sigma = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_enc = torch.nn.Parameter(torch.Tensor(hidden_size))
        self.b_miu = torch.nn.Parameter(torch.Tensor(hidden_size))
        self.b_sigma = torch.nn.Parameter(torch.Tensor(hidden_size))
        #decoder
        self.w_dec = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.w_x = torch.nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.w_z = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.w_out = torch.nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.b_z = torch.nn.Parameter(torch.Tensor(hidden_size))
        self.b_dec = torch.nn.Parameter(torch.Tensor(hidden_size))
        self.b_out = torch.nn.Parameter(torch.Tensor(input_size))

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, input, M, Delta):
        input = input.to(device).float()
        M = M.to(device).float()
        Delta = Delta.to(device).float()
        batchsize = input.size(0)
        step_size = input.size(1)

        listh_outs = []
        x_uncertainlys = []
        kldloss = 0
        #kldloss = kldloss.to(device)

        Delta = self.linearfordelta(Delta)
        Delta = 1 - Delta

        h1 = torch.zeros(batchsize, self.hidden_size).float().to(device)
        for i in range(step_size):
            delta = torch.squeeze(Delta[:, i:i + 1, :])  # batchsize,inputsize
            m = torch.squeeze(M[:, i:i + 1, :])
            x1 = torch.squeeze(input[:, i:i + 1, :])

            h1 = torch.tanh(torch.mm(h1, self.w_enc) + torch.mm(x1, self.w_in) + self.b_enc)

        miu_z = torch.mm(h1, self.w_miu) + self.b_miu
        sigma_z = torch.mm(h1, self.w_sigma) + self.b_sigma

        

        return



class GRU_Uncertainly(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super(GRU_Uncertainly, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.w_xr = torch.nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.w_hr = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.w_dr = torch.nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.w_mr = torch.nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.w_unr = torch.nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.w_zr = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.w_xz = torch.nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.w_hz = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.w_dz = torch.nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.w_mz = torch.nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.w_unz = torch.nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.w_zz = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.w_xh = torch.nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.w_hh = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.w_dh = torch.nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.w_mh = torch.nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.w_unh = torch.nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.w_zh = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size))

        self.b_z = torch.nn.Parameter(torch.Tensor(hidden_size))
        self.b_r = torch.nn.Parameter(torch.Tensor(hidden_size))
        self.b_h = torch.nn.Parameter(torch.Tensor(hidden_size))

        # 特征提取器
        self.phi_x = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU())
        self.phi_z = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU())

        #prior
        self.prior = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU())
        self.prior_mean = nn.Linear(hidden_size, hidden_size)
        self.prior_std = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus())

        # decoder
        self.dec = nn.Sequential(
            nn.Linear(hidden_size + hidden_size, hidden_size),
            nn.ReLU())
        self.dec_std = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.Softplus())
        self.dec_mean = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid())

        # encoder
        self.enc = nn.Sequential(
            nn.Linear(hidden_size + hidden_size, hidden_size),
            nn.ReLU())
        self.enc_mean = nn.Linear(hidden_size, hidden_size)
        self.enc_std = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus())


        self.reset_parameters()  # 初始化参数

        self.linearfordelta = nn.Sequential(
            nn.Sigmoid()
        )

        self.linearh2x = nn.Sequential(
            nn.Linear(hidden_size, input_size)
        )

        self.outlinear = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )  # 输出层

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, input, M, Delta):
        input = input.to(device).float()
        M = M.to(device).float()
        Delta = Delta.to(device).float()
        batchsize = input.size(0)
        step_size = input.size(1)

        listxs = []
        x_uncertainlys = []
        kldloss = 0
        #kldloss = kldloss.to(device)

        Delta = self.linearfordelta(Delta)
        Delta = 1 - Delta


        h1 = torch.zeros(batchsize, self.hidden_size).float().to(device)

        for i in range(step_size):
            delta = torch.squeeze(Delta[:, i:i + 1, :])  # batchsize,inputsize
            m = torch.squeeze(M[:, i:i + 1, :])
            x1 = torch.squeeze(input[:, i:i + 1, :])
            #print(delta.shape)
            #print(m.shape)
            #print(x1.shape)
            #流程是：x经过特征提取变成x1,x1和h[t-1]通过encoder生成隐变量z的均值与方差
            #使用重参数化得到z=方差*e+均值
            #先验prior(h)得到一个均值与方差
            #得到z以后用decoder得到x~的均值与方差
            #这时x~的均值就是补全的值，x~的方差就是它的不确定性
            #KL散度loss1使用encoder得到的z的均值方差，prior得到的均值与方差共同计算
            #补全的值与真实值之间的距离loss2使用MSE计算
            #标签分类loss3使用binary cross entropy计算

            #inference (xt,ht-1) -> z encoder
            if batchsize==1:
                delta = delta.unsqueeze(dim=0)
                m = m.unsqueeze(dim=0)
                x1 = x1.unsqueeze(dim=0)
            #特征提取
            phi_x_t = self.phi_x(x1)

            # encoder 得到隐变量z的均值与方差
            #print(phi_x_t.shape)
            #print(h1.shape)
            enc_t = self.enc(torch.cat((phi_x_t, h1), -1))
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t)

            # prior
            prior_t = self.prior(h1)
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # sampling and reparameterization
            z_eps = torch.randn_like(enc_std_t)
            phi_z_t = self.phi_z(enc_mean_t + z_eps * enc_std_t)

            # decoder 用decoder求出x的均值和方差
            dec_t = self.dec(torch.cat((phi_z_t, h1), -1))
            # 用这个当作补全x的值
            dec_mean_t = self.dec_mean(dec_t)
            # 这个相当于补全x的值的标准差
            dec_std_t = self.dec_std(dec_t)

            # recurrence
            #_, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

            #和delta一样，标准差越高，不确定性越高
            x_uncertainly = (1-m)*dec_std_t
            x_uncertainly = self.linearfordelta(x_uncertainly)
            x_uncertainly = x_uncertainly - 0.5 * torch.ones_like(x_uncertainly)

            x_uncertainlys.append(x_uncertainly)

            if i != 0:  # batchsize,inputsize
                x1 = m * x1 + (1 - m) * dec_mean_t * delta * x_uncertainly

            z = torch.sigmoid((torch.mm(x1, self.w_xz) + torch.mm(h1, self.w_hz) + torch.mm(delta,
                                                                                            self.w_dz) + torch.mm(m,
                                                                                                                  self.w_mz) + torch.mm(x_uncertainly,self.w_unz) + torch.mm(phi_z_t,self.w_zz) + self.b_z))  # batchsize,,hiddensize
            r = torch.sigmoid((torch.mm(x1, self.w_xr) + torch.mm(h1, self.w_hr) + torch.mm(delta,
                                                                                            self.w_dr) + torch.mm(m,
                                                                                                                  self.w_mr) + torch.mm(x_uncertainly,self.w_unr) + torch.mm(phi_z_t,self.w_zr) + self.b_r))  # batchsize,,hiddensize
            h1_tilde = torch.tanh(
                (torch.mm(x1, self.w_xh) + torch.mm(r * h1, self.w_hh) + torch.mm(delta, self.w_dh) + torch.mm(m,
                                                                                                               self.w_mh) + torch.mm(x_uncertainly,self.w_unh) + torch.mm(phi_z_t,self.w_zh) + self.b_h))  # batchsize,,hiddensize
            h1 = (1 - z) * h1 + z * h1_tilde  # batchsize,,hiddensize

            kldloss += kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)

            finalx = self.linearh2x(h1) # batchsize,inputsize

            listxs.append(finalx)

        xpreds = torch.stack(listxs).permute(1, 0, 2)  # batchsize,seqlen,inputsize seqlen是指第i个患者的时间步
        xuncer = torch.stack(x_uncertainlys).permute(1, 0, 2)

        #print(xuncer.shape)
        output = self.outlinear(h1)

        return output.squeeze(),xpreds,xuncer,kldloss

    def est_uncertainty(self, input, M, Delta, n_sample=25):
        # Monte Carlo
        samples = torch.zeros([n_sample, input.size(0)]).to(device)
        for i in range(n_sample):
            out, _, _, _ = self.forward(input, M, Delta)
            samples[i] = out

        sample_mean = samples.mean(0)
        sample_var = samples.var(0)
        return sample_mean, sample_var



def kld_gauss(mean_1, std_1, mean_2, std_2):
    """Using std to compute KLD"""
    kld_element = (2 * torch.log(std_2) - 2 * torch.log(std_1) +
                   (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
                   std_2.pow(2) - 1)
    return 0.5 * torch.sum(kld_element)


class My_mse_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, M, x, x_preds):
        return torch.mean(torch.pow((M * x - M * x_preds), 2))

model = GRU_Uncertainly(input_size, hidden_size)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



def train_model(model, train_loader, valid_loader):
    model.train()
    train_loss_array = []
    Early_stopping = Config.EarlyStopping()

    for epoch in range(epochs):
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        model.train()
        for i, data in enumerate(train_loader):
            # 这里的i是所有的traindata根据一个batchsize分成的总数，这里是119个
            # i是第i个batchsize
            # print(i)
            inputs, labels, m, delta, lens = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            m = m.to(device)
            delta = delta.to(device)
            lens = lens.to(device)
            labels = labels.float()

            # 前向传播
            out, xpreds, xunc, kldloss = model(inputs, m, delta)
            out = out.to(device)
            kldloss = kldloss.to(device)
            batch_loss = torch.tensor(0, dtype=float).to(device)
            # 二分类任务loss函数使用binary cross entropy，BCE
            lossF = torch.nn.BCELoss(size_average=True).to(device)
            # 得到loss分数
            batch_loss = lossF(out, labels)
            """
            for j in range(len(lens)):  # 遍历256个样本
                intlenj = int(lens[j])  # 这256个样本里的第j个样本的原本的长度
                # 我们要取的是out[:,intlenj-1,:],也就是最后一步的预测值，拿这个值和label比对
                oneloss = torch.tensor(0).to(device)
                lossF = torch.nn.BCELoss(size_average=True).to(device)
                # 下面是计算的一个样本的loss
                oneloss = lossF(out[j, intlenj - 1, :], labels[j].unsqueeze(dim=0))
                # print(oneloss)#这是一个样本的loss
                batch_loss += oneloss

            batch_loss /= batch_size
            """
            lossF2 = My_mse_loss()
            loss2 = lossF2(m, inputs, xpreds).to(device)

            losstotal = alpha_loss1*batch_loss + alpha_loss2*loss2 + alpha_loss3*kldloss

            #print("1"+str(alpha_loss1*batch_loss))
            #print("2"+str(alpha_loss2*loss2))
            #print("3"+str(alpha_loss3*kldloss))

            optimizer.zero_grad()
            losstotal.backward(retain_graph=True)
            optimizer.step()
        if epoch % 4 == 0:
            for p in optimizer.param_groups:
                p['lr'] *= 0.5
        if (epoch + 1) % 1 == 0:  # 每 1 次输出结果
            print('Epoch: {}, Train Loss: {}'.format(epoch + 1, losstotal.detach().data))
            train_loss_array.append(losstotal.detach().data)
            # 在验证集上过一遍
            device = torch.device("cpu")
            model.eval()
            valid_losses = []
            for i, data in enumerate(valid_loader):
                inputs, labels, m, delta, lens = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                m = m.to(device)
                delta = delta.to(device)
                lens = lens.to(device)
                labels = labels.float()
                # 前向传播
                out, xpreds, xunc, kldloss = model(inputs, m, delta)
                out = out.to(device)
                xpreds = xpreds.to(device)
                kldloss = kldloss.to(device)
                batch_loss = torch.tensor(0, dtype=float).to(device)
                # 二分类任务loss函数使用binary cross entropy，BCE
                lossF = torch.nn.BCELoss(size_average=True).to(device)
                # 得到loss分数
                batch_loss = lossF(out, labels)
                """
                for j in range(len(lens)):  # 遍历256个样本
                    intlenj = int(lens[j])  # 这256个样本里的第j个样本的原本的长度
                    lossF = torch.nn.BCELoss(size_average=True).to(device)
                    oneloss = lossF(out[j, intlenj - 1, :], labels[j].unsqueeze(dim=0)).to(device)
                    batch_loss += oneloss

                batch_loss /= batch_size
                """
                lossF2 = My_mse_loss()
                loss2 = lossF2(m, inputs, xpreds)

                losstotal = alpha_loss1*batch_loss + alpha_loss2*loss2 + alpha_loss3*kldloss
                # losstotal = uncertaintyloss(batch_loss, loss2)
                valid_losses.append(losstotal.detach().data)

            valid_loss = np.average(valid_losses)
            print('Epoch: {}, Valid Loss: {}'.format(epoch + 1, valid_loss))
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            Early_stopping(valid_loss, model, state, save_model_dir)

            if Early_stopping.early_stop:
                print("Early stopping")
                break


class My_rmse(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, M, x, x_preds):
        return torch.sqrt(torch.mean(torch.pow((M * x - M * x_preds), 2)))

class ResWithUncer:
    def __init__(self,res,label,uncer):
        self.res = res
        self.label = label
        self.uncer = uncer

#验证模型预测结果的不确定性
def valtestuncer(model, test_loader):
    device = torch.device("cpu")
    # model.eval()
    model.train()
    meantotallist = list()
    vartotallist = list()

    outs = list()
    labelss = list()

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels, m, delta, lens = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            m = m.to(device)
            delta = delta.to(device)
            lens = lens.to(device)
            labels = labels.float()

            # print(inputs.shape)

            # 前向传播
            out, xpreds, xunc, kldloss = model(inputs, m, delta)
            # print("模型预测的值："+str(xpreds[0]))
            # print("模型预测的不确定性："+str(xunc[0]))
            # print("真实值："+str(inputs[0]))
            #print(out)
            out = out.to(device)

            outs.extend(list(out.numpy()))
            labelss.extend(list(labels.numpy()))

            mean, var = model.est_uncertainty(inputs, m, delta)
            mean = mean.to(device)
            var = var.to(device)
            """
            print(mean.shape)
            print(var.shape)
            print("模型预测的值：")
            print(mean)
            print("每个值的不确定性：")
            print(var)"""

            #把均值，标签，不确定性分数放到对象里，按不确定性从高到低排序
            meanlist = mean.tolist()
            varlist = var.tolist()
            meantotallist.extend(meanlist)
            vartotallist.extend(varlist)

    #print(len(outs))
    #print(len(meantotallist))
    print(len(vartotallist))
    rwulist = list()
    total = len(meantotallist)
    #把数据放到对象里，对象放到list里
    for i in range(total):
        rwu = ResWithUncer(meantotallist[i],labelss[i],vartotallist[i])
        rwulist.append(rwu)
    #把list里的对象按不确定性分数从低到高排序
    rwulist.sort(key=lambda x: x.uncer, reverse=False)
    #for r in rwulist:
        #print(str(r.res)+" "+str(r.label)+" "+str(r.uncer))
    #依次去除不确定性分数从低到高排在前20%，20%。。。80%的数据，然后分别做auroc和auprc的统计
    #预计应该是数据越来越好
    per = 0.1

    for i in range(8):
        per = per + 0.1
        down = int(per * total)


        temprwulist = rwulist[down:]
        #print("down="+str(up)+" temp="+str(len(temprwulist)))
        tempouts = list()
        templabels = list()
        for r in temprwulist:
            tempouts.append(r.res)
            templabels.append(r.label)
        #print(len(tempouts))
        #print(len(templabels))
        nptempouts = np.array(tempouts)
        nptemplabels = np.array(templabels)
        #print(nptempouts)
        #print(nptemplabels)
        tempauroc = metrics.roc_auc_score(nptemplabels, nptempouts)
        (precisions, recalls, thresholds) = metrics.precision_recall_curve(nptemplabels, nptempouts)
        tempauprc = metrics.auc(recalls, precisions)
        print("去除不确定性分数排在前"+str(per*100)+"%的数据的auroc="+str(tempauroc)+" auprc="+str(tempauprc))

    """
    outs = np.array(outs)
    labelss = np.array(labelss)

    auroc = metrics.roc_auc_score(labelss, outs)

    (precisions, recalls, thresholds) = metrics.precision_recall_curve(labelss, outs)
    auprc = metrics.auc(recalls, precisions)

    return auroc, auprc
    """




#验证缺失值补全
def valtestcomplete(model,test_loader):
    device = torch.device("cpu")
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels, m, delta, lens = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            m = m.to(device)
            delta = delta.to(device)
            lens = lens.to(device)
            labels = labels.float()

            # print(inputs.shape)


            #print("模型预测的值："+str(xpreds[0]))
            #print("模型预测的不确定性："+str(xunc[0]))
            #print("真实值："+str(inputs[0]))
            #print(xpreds.shape) #256,35,63
            #print(xunc.shape)
            #print(inputs.shape)
            listinputspermute = inputs.permute(0, 2, 1).numpy().tolist()
            listmpermute = m.permute(0, 2, 1).numpy().tolist()  # batch,inputsize,seq
            percentage_batchsize = []
            jks = []
            for j in range(256):
                percentage_feature = []
                for k in range(63):
                    percentage = 1 - (listmpermute[j][k][:lens[j]].count(0) / lens[j])  # 统计每个患者每个特征在所有时间步内非0的比例
                    # print(percentage)
                    if percentage < 1 and percentage > 0.8 and lens[j] > 20:
                        per = percentage.item()
                        jks.append(str(j) + "/" + str(k) + "/" + str(per) + "/" + str(int(lens[j] * 0.1)))
                        print("第" + str(j) + "个患者的第" + str(k) + "个特征的非0比例超过0.8小于1总步长大于20，为" + str(per) + ",最终" + str(
                            labels[j]))
                    percentage_feature.append(percentage)
                percentage_batchsize.append(percentage_feature)
            # print(percentage_batchsize)
            # print(percentage_batchsize[5])
            x = list(range(35))
            for jk in jks:
                j = int(jk.split("/")[0])
                k = int(jk.split("/")[1])
                per = float(jk.split("/")[2])

                deletepoints = int(jk.split("/")[3])  # 要遮蔽的点的个数
                print("第" + str(j) + "个患者的第" + str(k) + "个特征的非0比例超过0.8小于1总步长大于20，为" + str(per) + ",最终" + str(
                    labels[j]) + ",要遮蔽的点的个数是" + str(deletepoints))
                point = 0 #已遮蔽的点的个数

                mpermute = m[j].permute(1, 0)  # 转置成inputsize,seqlen
                inputspermute = inputs[j].permute(1, 0) # 转置成inputsize,seqlen

                temprand = random.randint(0, lens[j] - 1)  # 从这个真实步长中选随机选一个点遮蔽

                randsteps = []  # 记下随机遮蔽的时间步位置
                trues = list(range(lens[j]))  # 记下遮蔽的真实值
                truesandsteps = []  # 记下遮蔽的真实值/时间步位置

                while point<deletepoints: #当已遮蔽的点的个数<要遮蔽的点的个数时
                    while mpermute[k][temprand]==0:#如果选的点已经是缺失值就重选
                        temprand = random.randint(0,lens[j]-1)
                    print("遮蔽的真实值"+str(inputspermute[k][temprand].item()))
                    print("随机遮蔽的时间步位置" + str(temprand))
                    truesandsteps.append(str(inputspermute[k][temprand].item())+"/"+str(temprand))
                    trues[temprand] = inputspermute[k][temprand].item() # 把真实值放在trues里面
                    inputspermute[k][temprand] = 0 #把真实值置0，代表遮蔽了
                    mpermute[k][temprand]=0 #m也置0

                    randsteps.append(temprand) #把遮蔽的位置下标加进去

                    point = point+1 #已遮蔽点数加1

                inputsone = inputspermute.permute(1, 0) #遮蔽完以后再转置回来
                mone = mpermute.permute(1, 0) #遮蔽完以后再转置回来
                #print(mone.unsqueeze(dim=0).shape) #1,35,63
                #print(inputsone.unsqueeze(dim=0).shape) #1,35,63
                #预测
                out, xpreds, xunc, kldloss = model(inputsone.unsqueeze(dim=0), mone.unsqueeze(dim=0), delta[j].unsqueeze(dim=0))
                #print(xpreds.shape) #1,35,63
                #print(xunc.shape) #1,35,63
                xpreds = xpreds.squeeze()
                xunc = xunc.squeeze()

                #print(xpreds.shape)  # 35,63
                #print(xunc.shape)  # 35,63
                listxpreds = xpreds.to(device).numpy().tolist()
                listxunc = xunc.to(device).numpy().tolist()
                listinput = inputsone.to(device).numpy().tolist()
                #print(listxpreds)
                #print(listxunc)
                y1 = []  # 预测值点
                inputy = [] #真实值点
                yxunc0 = xpreds.to(device).numpy()
                yxunc = xunc.to(device).numpy()
                yxuncup = yxunc0 + yxunc
                yxuncdown = yxunc0 - yxunc
                listyxuncup = yxuncup.tolist()
                listyxuncdown = yxuncdown.tolist()
                listmone = mone.to(device).numpy().tolist()
                plotyxuncup = [] # 不确定性上限
                plotyxuncdown = [] #不确定性下限
                mm = [] #mask矩阵
                for time in range(35):
                    y1.append(listxpreds[time][k])
                    inputy.append(listinput[time][k])
                    plotyxuncup.append(listyxuncup[time][k])
                    plotyxuncdown.append(listyxuncdown[time][k])
                    mm.append(listmone[time][k])
                #明确画图需要什么数据
                #1.xpreds，画出模型预测的折线图
                #2.randsteps 这个数组中存放的是这个折线中遮蔽的点的位置下标
                #3.trues[randsteps[0..i]] 这个数组中存放的是遮蔽的点的真实值
                #4.xunc 这个存放的是模型预测的值的不确定性
                #需要把数据变成一维数组list，横坐标是x = list(range(lens[j])) 纵坐标是这些点的数组
                x = list(range(lens[j]))
                print(y1)
                print(plotyxuncup)
                print(plotyxuncdown)
                print(mm)
                plt.clf()
                plt.plot(x, y1[:lens[j]], color="r", label="predict", marker="o")
                plt.plot(x, inputy[:lens[j]], color="y", label="origin", marker="o")
                plt.fill_between(x, plotyxuncdown[:lens[j]], plotyxuncup[:lens[j]], color='cornflowerblue')

                flag = False
                flag2 = False
                for time in range(lens[j]):
                    if mm[time] == 0:
                        if flag:
                            plt.scatter(x[time], inputy[time], color='springgreen', marker='o', edgecolors='deeppink',
                                        s=150)
                        else:
                            plt.scatter(x[time], inputy[time], color='springgreen', marker='o', edgecolors='deeppink',
                                        s=150,label="origin-mask")
                            flag = True
                        if time in randsteps:  # 如果这个点是遮蔽点,就输出它的真实点,并且画出预测值和真实点的虚线
                            if flag2:
                                plt.scatter(x[time], trues[time], color='saddlebrown', marker='*', edgecolors='g', s=200)
                            else:
                                plt.scatter(x[time], trues[time], color='saddlebrown', marker='*', edgecolors='g', s=200, label="origin-true")
                                flag2 = True
                            if trues[time] < y1[time]:
                                plt.vlines(x=x[time], ymin=trues[time], ymax=y1[time], colors='purple',
                                           linestyles="dashed")
                            else:
                                plt.vlines(x=x[time], ymin=y1[time], ymax=trues[time], colors='purple',
                                           linestyles="dashed")


                plt.xticks(rotation=45)
                plt.xlabel("timestamps")
                plt.ylabel("y_nums")
                plt.title(str(i+1) + "_" + str(j + 1) + "_" + str(k + 1) + ".jpg")
                plt.legend(loc="upper left")
                plt.savefig("imgs2/" + str(i+1) + "_" + str(j + 1) + "_" + str(k + 1) + "kkk.png")
                #plt.show()


                #break
                #xpreds画出的折线，把其中m=0的点都弄成空心的
                #然后把真实的值点在图中画成叉号x
                #然后把空心的点对应的xunc值，弄成阴影plt.fill_between(x,y1,y2,color='b')



            #break








def test_model(model, test_loader):
    device = torch.device("cpu")
    #model.eval()
    test_loss_array = []
    test_loss_array2 = []
    outs = list()
    labelss = list()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels, m, delta, lens = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            m = m.to(device)
            delta = delta.to(device)
            lens = lens.to(device)
            labels = labels.float()

            # print(inputs.shape)


            # 前向传播
            out, xpreds, xunc, kldloss = model(inputs, m, delta)
            #print("模型预测的值："+str(xpreds[0]))
            #print("模型预测的不确定性："+str(xunc[0]))
            #print("真实值："+str(inputs[0]))
            print(out)
            out = out.to(device)

            """
            mean,var = model.est_uncertainty(inputs,m,delta)
            mean = mean.to(device)
            var = var.to(device)
            print(mean.shape)
            print(var.shape)
            print("模型预测的值：")
            print(mean)
            print("每个值的不确定性：")
            print(var)"""

            batch_loss = torch.tensor(0, dtype=float).to(device)
            # 二分类任务loss函数使用binary cross entropy，BCE
            lossF = torch.nn.BCELoss(size_average=True).to(device)
            # 得到loss分数
            batch_loss = lossF(out, labels)
            """
            for j in range(len(lens)):  # 遍历256个样本
                intlenj = int(lens[j])  # 这256个样本里的第j个样本的原本的长度
                oneloss = torch.tensor(0).to(device)
                lossF = torch.nn.BCELoss(size_average=True).to(device)
                # 下面是计算的一个样本的loss
                oneloss = lossF(out[j, intlenj - 1, :], labels[j].unsqueeze(dim=0))
                outs.extend(list(out[j, intlenj - 1, :].numpy()))
                # print(list(out[j, intlenj - 1, :].numpy()))
                templabel = [int(labels[j])]
                labelss.extend(templabel)
                batch_loss += oneloss

            batch_loss /= batch_size
            """
            outs.extend(list(out.numpy()))
            labelss.extend(list(labels.numpy()))
            lossF = torch.nn.BCELoss(size_average=True).to(device)
            # loss = lossF(out, labels)
            xpreds = xpreds.to(device)
            lossF2 = My_mse_loss().to(device)
            loss2 = lossF2(m, inputs, xpreds)

            #print('Test loss1:{}'.format(float(batch_loss.data)))
            test_loss_array.append(float(batch_loss.data))
            #print('Test loss2:{}'.format(float(loss2.data)))
            #print('Test loss3:{}'.format(float(kldloss)))
            test_loss_array2.append(float(loss2.data))

    outs = np.array(outs)
    labelss = np.array(labelss)

    auroc = metrics.roc_auc_score(labelss, outs)

    (precisions, recalls, thresholds) = metrics.precision_recall_curve(labelss, outs)
    auprc = metrics.auc(recalls, precisions)

    return auroc, auprc


def mask10perdata(model):
    x_torch = pickle.load(open('dataset/lb_' + datatype + '_x_for_missingvalue.p', 'rb'))
    m_torch = pickle.load(open('dataset/lb_' + datatype + '_m_for_missingvalue.p', 'rb'))

    delta_torch = pickle.load(open('dataset/lb_' + datatype + '_delta_for_missingvalue.p', 'rb'))
    y_torch = pickle.load(open('dataset/lb_' + datatype + '_y.p', 'rb'))
    x_lens = pickle.load(open('dataset/lb_' + datatype + '_len.p', 'rb'))

    # 划分训练集验证集和测试集，8：1：1
    train_ratio = 0.8
    valid_ratio = 0.1
    test_ratio = 0.1

    N = len(x_torch)
    testing_x = x_torch[int((train_ratio + valid_ratio) * N):]
    testing_m = m_torch[int((train_ratio + valid_ratio) * N):]
    testing_delta = delta_torch[int((train_ratio + valid_ratio) * N):]
    testing_x_lens = x_lens[int((train_ratio + valid_ratio) * N):]
    testing_y = y_torch[int((train_ratio + valid_ratio) * N):]

    N = len(testing_m)  # 4629
    # indices = np.arange(256 * 63 * 35)  # 遮蔽一个患者的10%的数据
    # val_indices = np.random.choice(indices, 63 * 35 // 10, replace=False).tolist()
    print(N)
    # for j in range(N):

    m = torch.reshape(testing_m, (-1, N * 63 * 35)).squeeze()  # 4629,2205 因为np.random.choice只能处理一维数组
    timestepidx = []  # 把m=1的点都挑出来的下标列表
    timestepidx = pickle.load(open('lb_' + datatype + '_maskidx.p', 'rb'))

    print(len(timestepidx))  # 1133452个点是有值的

    timestep = torch.zeros_like(m)  # 把m=1的点都挑出来的矩阵,随机遮蔽的点的位置最后置1 4629,2205

    # timestep = torch.reshape(timestep, (-1, N * 63 * 35)).squeeze()#因为np.random.choice只能处理一维数组
    print(timestep.shape)
    # timestep.numpy()  # list转numpy

    masktimestepidx = np.random.choice(timestepidx, len(timestepidx) // 10,
                                       replace=False).tolist()  # 从m=1的下标列表这里面随机选出10%的要遮蔽的点的下标
    print(len(masktimestepidx))
    for i in masktimestepidx:
        timestep[i] = 1  # 遮蔽这些点，遮蔽的点置1

    m = m - timestep  # 1-1=0，这就实现了把m矩阵对应的点变为0的工作
    # 把m和timestep还原成三维，
    m = torch.reshape(m, (N, 35, 63))
    timestep = torch.reshape(timestep, (N, 35, 63))
    testing_x_mask = (1 - timestep) * testing_x  # 同时也实现了把输入对应的点变为0的工作
    # testing_x是真实值输入矩阵，testing_x_mask是遮蔽之后对应的值变为0的矩阵
    print(timestep.shape)
    print(testing_x.shape)
    print(m.shape)

    test_deal_dataset = TensorDataset(testing_x, testing_x_mask, testing_y, m, testing_delta, testing_x_lens, timestep)

    test_loader = DataLoader(dataset=test_deal_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             drop_last=True,
                             num_workers=workers)

    device = torch.device("cpu")
    model.eval()
    k = 0
    totalrmse = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, inputsmask, labels, m, delta, lens, timestepbatch = data
            inputs = inputs.to(device)
            inputsmask = inputsmask.to(device)
            labels = labels.to(device)
            m = m.to(device)
            delta = delta.to(device)
            lens = lens.to(device)
            labels = labels.float()

            # print(inputs.shape)
            # 前向传播
            out, xpreds, xunc, kldloss = model(inputsmask, m, delta)

            xpreds = xpreds.to(device)

            rmsefunction = My_rmse().to(device)
            rmse = rmsefunction(timestepbatch, inputs, xpreds)
            totalrmse = totalrmse + rmse.item()
            k = k + 1

    finalrmse = float(totalrmse / k)
    print("rmse="+str(finalrmse));
    return finalrmse

def main():
    train_loader, test_loader, valid_loader = get_data()
    #train_model(model, train_loader, valid_loader)
    # 加载保存的模型直接进行测试机验证，不进行此模块以后的步骤

    checkpoint = torch.load(save_model_dir)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epochs = checkpoint['epoch']

    """
    aurocs = []
    auprcs = []
    rmses = []
    for i in range(testepoch):
        _, test_loader, _ = get_data()
        auroc, auprc = test_model(model, test_loader)
        aurocs.append(auroc)
        auprcs.append(auprc)

    auroc_mean = np.mean(aurocs)
    auroc_std = np.std(aurocs, ddof=1)
    auprc_mean = np.mean(auprcs)
    auprc_std = np.std(auprcs, ddof=1)
    print("auroc 平均值为：" + str(auroc_mean) + " 标准差为：" + str(auroc_std))
    print("auprc 平均值为：" + str(auprc_mean) + " 标准差为：" + str(auprc_std))
    
    # printimg(model,test_loader)
    mask10perdata(model)

    # printimg(model, test_loader)
    """
    _, test_loader, _ = get_data()
    valtestuncer(model, test_loader)
    #valtestcomplete(model,test_loader)

    return


if __name__ == '__main__':
    main()

