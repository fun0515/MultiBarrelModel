import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from torch.utils import data
from dataset import tactileDataset
from utils import mem_update_adp, Conv1dLinear
from MultiBarrel4EvTask import TemporalCompressor, count_params

class Readout1D(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=512, num_classes=36):
        super().__init__()
        # 2D卷积
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, 11),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )

        # 分类器（添加LayerNorm）
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        """输入x: (B, T=250, N=39, D=64) → 调整为 (B, D, T, N)"""
        x = x.permute(0, 3, 1, 2).squeeze()  # (B, D, T)
        x = self.conv(x)  # (B,hidden_dim,T,39)
        avg_pool = x.mean(dim=[2]) # (B, hidden_dim)
        return self.classifier(avg_pool) # (B, num_classes)

class SingleBarrelModel(nn.Module):
    def __init__(self, input_size=2, output_size=3, init_b=0.02, init_tau_a = 5, init_tau_m = 10,):
        super(SingleBarrelModel, self).__init__()
        self.output_size, self.init_b, self.init_tau_a, self.init_tau_m, = output_size, init_b, init_tau_a, init_tau_m,
        self.num_barrels = 1
        global b_j0
        b_j0 = init_b

        self.popSize, self.popType, self.ThtoAll, self.AlltoAll = self.readPops()
        self.compressor = TemporalCompressor(in_channels=input_size, out_channels=64, compressed_T=64, stride=2)
        # 生成丘脑到bfd的连接
        self.ThtoPops = nn.ModuleList()  # 丘脑只对其中的11个群落有兴奋型连接
        for i, conn_prob in enumerate(self.ThtoAll):
            if conn_prob != 0.:
                self.ThtoPops.append(Conv1dLinear(num_barrels=self.num_barrels,input_size=39*64, output_size=self.popSize[i]))

        # 生成神经元群落之间连接
        self.toPops = nn.ModuleList()
        for i, size in enumerate(self.popSize):
            inner_list = nn.ModuleList()
            for j, conn_prob in enumerate(self.AlltoAll[:,i]):
                if conn_prob != 0. :  # 不排除群落内部连接
                    inner_list.append(Conv1dLinear(num_barrels=self.num_barrels, input_size=self.popSize[j], output_size=size))
            self.toPops.append(inner_list)

        # 生成动力学参数
        self.tau_adp, self.tau_m = nn.ParameterList(), nn.ParameterList()
        for size in self.popSize:
            self.tau_adp.append(nn.Parameter(torch.Tensor(self.num_barrels, size).cuda(), requires_grad=True))
            self.tau_m.append(nn.Parameter(torch.Tensor(self.num_barrels, size).cuda(), requires_grad=True))

        # readout
        self.readout = Readout1D(in_dim=self.popSize[0] * 4, num_classes=output_size)
        self.init_parameters()

    def init_parameters(self):
        for i in range(len(self.popSize)):
            nn.init.normal_(self.tau_adp[i], mean=self.init_tau_a, std=1.0)
            nn.init.normal_(self.tau_m[i], mean=self.init_tau_m, std=1.0)

    def get_all_params(self):
        params_base = []
        for module in self.toPops:
            params_base.extend(list(module.parameters()))
        params_base.extend(list(self.ThtoPops.parameters()) +
                           list(self.compressor.parameters()) + list(self.readout.parameters()))
        params_tau_adp = self.tau_adp
        params_tau_m = self.tau_m
        return params_base, params_tau_adp, params_tau_m

    def readPops(self):
        popType = ['E', 'E', 'E', 'E', 'E', 'E', 'E', 'E']
        popName = ['L2Pyr', 'L3Pyr', 'SSPyr', 'Pyr', 'STPyr', 'TTPyr', 'CTPyr', 'CCPyr']
        popSize = [32*39]*8
        Exc_ThtoAll = np.array([1, 1, 1., 1., 1, 1, 1, 1])
        Type_AlltoAll = np.array([
            [1., 1., 0., 0., 1., 0., 0., 0.],
            [1., 1., 1., 1., 0., 1., 0., 0.],
            [1., 1., 1., 1., 0., 0., 1., 0.],
            [1., 1., 1., 1., 0., 0., 1., 0.],
            [1., 1., 0., 0., 1., 0., 1., 0.],
            [0., 0., 1., 1., 0., 1., 1., 0.],
            [0., 0., 1., 1., 1., 0., 1., 1.],
            [0., 0., 1., 1., 1., 1., 1., 1.]])
        return popSize, popType, Exc_ThtoAll, Type_AlltoAll

    def forward(self, inputs):
        inputs = self.compressor(inputs)  # (B,N,2,250) -> (B,N,64,T)
        batch_size, sensor_num, channel, seq_num = inputs.size()
        inputs = inputs.view(batch_size, 1, sensor_num * channel, seq_num) # (B, 1, 39*2, T)

        mems = [torch.rand(batch_size, self.num_barrels, size).cuda() for size in self.popSize]
        spikes = [torch.rand(batch_size, self.num_barrels, size).cuda() for size in self.popSize]
        self.b = [(torch.ones((self.num_barrels, 1)) * self.init_b).cuda() for _ in range(len(self.popSize))]

        L56_state, h_state, m_state = [], [], []
        pre_mem_state = [torch.zeros(batch_size, self.num_barrels, size).cuda() for size in self.popSize]

        for t in range(seq_num):
            input = inputs[:,:,:,t]
            # 更新各群落状态
            for i in range(len(self.popSize)):
                input_current = 0.
                # 计算丘脑对第i个群落的输入(若存在)
                Th2pop_id = np.where(self.ThtoAll != 0.)[0]
                if i in Th2pop_id:
                    input_current = self.ThtoPops[np.where(Th2pop_id == i)[0][0]](input)

                # 计算群落之间输入
                source_index = np.where(self.AlltoAll[:, i] != 0.)[0]

                for j in range(len(self.toPops[i])):
                    input_current = input_current + self.toPops[i][j](spikes[source_index[j]])

                # 更新神经元状态
                mems[i], spikes[i], B, self.b[i] = mem_update_adp(input_current, mems[i], spikes[i], self.tau_adp[i], self.b[i], self.tau_m[i])
                pre_mem_state[i] = mems[i] + spikes[i] * B

            L56_state.append(torch.cat((spikes[4], spikes[5], spikes[6], spikes[7]), dim=-1))  # (B,39,2*popsize)
            h_state.append(torch.cat((spikes), dim=1))
            m_state.append(torch.cat((pre_mem_state), dim=1))
        L56_state = torch.stack(L56_state, dim=1)  # (B,T,1,32*4)
        output = self.readout(L56_state)
        return output, h_state, m_state

def train(init_b, init_tau_a, init_tau_m, batch_size=1024):
    num_epochs = 500
    input_dim, output_dim, seq_dim = 2, 36, 250

    train_dataset = tactileDataset(r'./data/Ev-Objects', train=True)
    test_dataset = tactileDataset(r'./data/Ev-Objects', train=False)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = SingleBarrelModel(input_size=input_dim, output_size=output_dim, init_b=init_b, init_tau_a=init_tau_a, init_tau_m=init_tau_m)
    count_params(model)
    device = torch.device('cuda')
    print("device:", device)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.0008

    params_base, params_tau_adp, params_tau_m = model.get_all_params()
    optimizer = torch.optim.AdamW([
        {'params': params_base, 'lr': learning_rate},
        {'params': params_tau_adp, 'lr': learning_rate * 1},
        {'params': params_tau_m, 'lr': learning_rate * 1}],weight_decay=0.1)

    scheduler = StepLR(optimizer, step_size=10, gamma=.8)

    best_accuracy = 0
    train_accs = []
    test_accs = []
    class_correct = [0] * output_dim  # 初始化每个类别的正确预测数
    class_total = [0] * output_dim  # 初始化每个类别的总样本数
    for epoch in range(num_epochs):
        train_correct = 0
        train_total = 0
        for inputs, labels in tqdm(train_loader):
            labels = labels.long().squeeze().to(device)
            optimizer.zero_grad()
            outputs, _, _ = model(inputs.cuda())
            loss = criterion(outputs, labels)
            loss.backward()

            # Updating parameters
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted.cpu() == labels.long().cpu()).sum()
        train_acc = 100. * train_correct.numpy() / train_total
        train_accs.append(train_acc)
        print('epoch: ', epoch, '. Loss: ', loss.item(), '. Train Acc: ', train_acc)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            test_correct = 0
            test_total = 0
            for inputs, labels in tqdm(test_loader):
                outputs, _, _ = model(inputs.cuda())
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted.cpu() == labels.long().squeeze().cpu()).sum()

                # 计算每个类别的正确预测
                for true, pred in zip(labels, predicted):
                    target = true.item()
                    class_correct[target] += (pred == target).item()
                    class_total[target] += 1

            test_acc = 100. * test_correct.numpy() / test_total
        test_accs.append(test_acc)
        if test_acc >= best_accuracy:
            torch.save(model.state_dict(), './SingleBarrelModel_Ev-Objects.pth')
            best_accuracy = test_acc
        print('epoch: ', epoch, '. Test Acc: ', test_acc, '. Best Acc: ', best_accuracy)

    return best_accuracy

if __name__ == '__main__':
    acc = train(init_b=0.46028508, init_tau_a=9.90546331, init_tau_m=4.34378016, batch_size=600)
    print(acc)