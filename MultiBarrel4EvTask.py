import os

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from torch.utils import data
import torch.nn.functional as F
from dataset import tactileDataset
from utils import mem_update_adp, mem_update, Conv1dLinear


class TemporalCompressor(nn.Module):
    def __init__(self,in_channels = 2, out_channels = 64, compressed_T = 124, stride = 2):
        super().__init__()
        self.out_channels = out_channels
        # 主卷积路径
        self.conv = nn.Sequential(
            # 第一层：通道扩展 + 时间压缩
            nn.Conv1d(in_channels=in_channels,out_channels=out_channels,kernel_size=21, dilation=3, stride=2,
            ),  # (B*39, out_channels, (T + padding*2 - kernel_size)//stride + 1)
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=21, dilation=3, stride=1,
                      ),  # (B*39, out_channels, (T + padding*2 - kernel_size)//stride + 1)
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x):
        B, N, C, T = x.size()
        x = x.reshape(B * N, C, T)  # (B*N, C, T)
        out = self.conv(x)# (B*N, out_channels, compressed_T)
        return out.view(B, N, self.out_channels, -1)

class Readout(nn.Module):
    def __init__(self, T=250, in_dim=128, hidden_dim=512, num_classes=36):
        super().__init__()
        # 2D卷积
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, (11, 39)),
            nn.BatchNorm2d(hidden_dim),
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
        x = x.permute(0, 3, 1, 2)  # (B, D, T, N)
        x = self.conv(x)  # (B,hidden_dim,T,39)
        avg_pool = x.mean(dim=[2, 3]) # (B, hidden_dim)
        return self.classifier(avg_pool) # (B, num_classes)

class NeighborAggregator(nn.Module):
    def __init__(self, k=2, popSize = 32, select_ratio=0.2):
        super().__init__()
        self.k, self.popSize = k, popSize
        self.select_num = int(k * 8 * select_ratio)
        self.neighbor_indices = self.compute_neighbors(k=k) # (N,K)

        # 门控网络
        self.gate_net = nn.Sequential(
            nn.Linear(64, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, k*8)) # (B, 39, K*8))


        # 预定义邻域聚合器（参数共享）
        self.aggregator = nn.Sequential(
            nn.Linear(self.select_num * popSize, popSize),
            nn.LayerNorm(popSize))
    def compute_neighbors(self,k=4):
        """返回: (N, K)邻域索引"""
        coords = torch.tensor([[-6, 0], [-5.3, -3], [-5.3, 3], [-4.6, -7.8], [-4.6, 7.8],
                           [-3.5, 0], [-3.05, -5.2], [-3.05, 5.2], [-3.1, -1.75], [-3.1, 1.75],  # 10
                           [-1.6, -8.9], [-1.75, -3], [-1.6, 8.9], [-1.75, 3], [-1.5, 0],
                           [-0.7, -1.3], [-0.7, 1.3], [0, -6], [0, 6], [0, -3.5],
                           [0, 0], [0, 3.5], [0.8, -1.3], [0.8, 1.3], [1.6, -8.9],
                           [1.6, 8.9], [1.5, 0], [1.75, 3], [1.75, -3], [3.05, -5.2],
                           [3.1, 1.75], [3.05, 5.2], [3.1, -1.75], [3.6, 0], [4.6, -7.8],
                           [4.6, 7.8], [5.3, -3], [5.3, 3], [6, 0]
                           ])
        dist_matrix = torch.cdist(coords, coords)  # (N, N)
        return torch.topk(dist_matrix, k+1, largest=False)[1][:, 1:k+1] # (N, K)

    def forward(self, x, spikes):
        """x: 当前时刻输入 (B, N, 64), spikes: 所有桶内神经元状态 (B, N, 256). 输出: 邻域聚合后的特征 (B, N, D)"""
        B,N,_ = spikes.size()

        gate_weights = self.gate_net(x.view(B*N,-1)).view(B,N,-1)  # 计算邻居群落权重 (B,39,K * 8)
        neighbor_feats = spikes[:, self.neighbor_indices, :].view(B,N,self.k*8,self.popSize)  # 收集邻域特征,(B, N, K, D) -> (B,N,K*8,32)
        _, topk_indices = torch.topk(gate_weights, self.select_num, dim=-1)  # 选择前几的神经元群落 (B,39,select_num)
        selected_feats = neighbor_feats.gather(2, topk_indices.unsqueeze(-1).expand(-1, -1, -1, 32)) # (B,39,select_num,32)
        output = self.aggregator(selected_feats.view(B,N,-1)) # 输出桶间电流 (B, N, popSize)
        return output

class MultiBarrelModel(nn.Module):
    def __init__(self, num_barrels=39, input_size=2, output_size=36, init_b=0.02, init_tau_a = 5, init_tau_m = 10):
        super(MultiBarrelModel, self).__init__()
        global b_j0
        b_j0 = init_b
        # 初始化多桶模型
        self.num_barrels = num_barrels
        self.output_size = output_size
        self.init_b, self.init_tau_a, self.init_tau_m = init_b, init_tau_a, init_tau_m

        self.popSize, self.popType, self.ThtoAll, self.AlltoAll = self.readPops()

        self.compressor = TemporalCompressor(in_channels=input_size, out_channels=64, compressed_T=64, stride=2)

        # 生成丘脑到bfd的连接
        self.ThtoPops = nn.ModuleList()  # 丘脑只对其中的11个群落有兴奋型连接
        for i, conn_prob in enumerate(self.ThtoAll):
            if conn_prob != 0.:
                self.ThtoPops.append(Conv1dLinear(num_barrels=self.num_barrels,input_size=64, output_size=self.popSize[i]))

        # 生成神经元群落之间连接
        self.toPops = nn.ModuleList()
        for i, size in enumerate(self.popSize):
            inner_list = nn.ModuleList()
            for j, conn_prob in enumerate(self.AlltoAll[:, i]):
                if conn_prob != 0.:  # 不排除群落内部连接
                    inner_list.append(
                        Conv1dLinear(num_barrels=self.num_barrels, input_size=self.popSize[j], output_size=size))
            self.toPops.append(inner_list)

        # 生成动力学参数
        self.tau_adp, self.tau_m = nn.ParameterList(), nn.ParameterList()
        for size in self.popSize:
            self.tau_adp.append(nn.Parameter(torch.Tensor(self.num_barrels, size).cuda(), requires_grad=True))
            self.tau_m.append(nn.Parameter(torch.Tensor(self.num_barrels, size).cuda(), requires_grad=True))

        # 生成桶间连接
        self.neighbor_agg = NeighborAggregator(k=4)

        # readout
        self.readout = Readout(T=64, in_dim=32*4, num_classes=output_size)
        self.init_parameters()
    def init_parameters(self):
        for i in range(len(self.popSize)):
            nn.init.normal_(self.tau_adp[i], mean=self.init_tau_a, std=1.0)
            nn.init.normal_(self.tau_m[i], mean=self.init_tau_m, std=1.0)
            # nn.init.constant_(self.tau_adp[i], self.init_tau_a)
            # nn.init.constant_(self.tau_m[i], self.init_tau_m)

    def readPops(self):
        popType = ['E', 'E', 'E', 'E', 'E', 'E', 'E', 'E']
        popName = ['L2Pyr', 'L3Pyr', 'SSPyr', 'Pyr', 'STPyr', 'TTPyr', 'CTPyr', 'CCPyr']
        popSize = [32]*8
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
    def get_all_params(self):
        params_base = []
        for module in self.toPops:
            params_base.extend(list(module.parameters()))
        params_base.extend(list(self.ThtoPops.parameters()) + list(self.compressor.parameters()) +
                           list(self.readout.parameters()) + list(self.neighbor_agg.parameters()))
        return params_base, self.tau_adp, self.tau_m


    def forward(self, inputs):
        inputs = self.compressor(inputs) # (B,N,2,250) -> (B,N,64,T')
        batch_size, sensor_num, channel, seq_num = inputs.size()  #

        mems = [torch.rand(batch_size, self.num_barrels, size).cuda() for size in self.popSize]
        spikes = [torch.rand(batch_size, self.num_barrels, size).cuda() for size in self.popSize]
        self.b = [(torch.rand((self.num_barrels, 1)) * self.init_b).cuda() for _ in range(len(self.popSize))]

        L56_state, h_state, m_state = [], [], []
        pre_mem_state = [torch.zeros(batch_size, self.num_barrels, size).cuda() for size in self.popSize]
        for t in range(seq_num):
            input = inputs[:, :, :, t]  # (B, N_barrels, 64)

            # 更新各群落状态
            for i in range(len(self.popSize)):
                input_current = 0.
                # 计算丘脑对第i个群落的输入(若存在)
                Th2pop_id = np.where(self.ThtoAll != 0.)[0]
                if i in Th2pop_id:
                    input_current += self.ThtoPops[np.where(Th2pop_id == i)[0][0]](input)

                # 计算其他神经元群落对群落i的输入
                source_index = np.where(self.AlltoAll[:, i] != 0.)[0]
                for j in range(len(self.toPops[i])):
                    input_current += self.toPops[i][j](spikes[source_index[j]])

                # 计算邻居桶输入
                input_current += self.neighbor_agg(x=input,spikes=torch.cat(spikes,dim=-1))

                # 更新神经元状态
                mems[i], spikes[i], B, self.b[i] = mem_update_adp(input_current, mems[i], spikes[i], self.tau_adp[i],
                                                                  self.b[i], self.tau_m[i])

                pre_mem_state[i] = mems[i] + spikes[i] * B
            h_state.append(torch.cat(spikes,dim=-1)) # (B, N_barrels, N_neurons)
            L56_state.append(torch.cat((spikes[4],spikes[5], spikes[6],spikes[7]),dim=-1)) # (B,39,2*popsize)
            m_state.append(torch.cat((pre_mem_state), dim=-1))
        h_state = torch.stack(h_state,dim=1) # (B, T, N_barrels, N_neurons)
        m_state = torch.stack(m_state, dim=1)  # (B, T, N_barrels, N_neurons)
        L56_state = torch.stack(L56_state,dim=1) # (B,T,39,2*popsize)
        output = self.readout(L56_state)
        return F.softmax(output,dim=1), h_state, m_state

def count_params(model):
    total = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num = param.numel()
            total += num
            print(f"{name}: {num}")
    print('total params:', total)
    return total

def train(init_b, init_tau_a, init_tau_m,batch_size=600):
    num_epochs = 200
    input_dim, output_dim = 2, 20

    train_dataset = tactileDataset(r'./data/Ev-Containers', train=True)
    test_dataset = tactileDataset(r'./data/Ev-Containers', train=False)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = MultiBarrelModel(num_barrels=39, output_size=output_dim, init_b=init_b, init_tau_a=init_tau_a, init_tau_m=init_tau_m)
    count_params(model)
    device = torch.device('cuda')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.0008

    params_base, params_tau_adp, params_tau_m = model.get_all_params()
    optimizer = torch.optim.AdamW([
        {'params': params_base, 'lr': learning_rate},
        {'params': params_tau_adp, 'lr': learning_rate * 1},
        {'params': params_tau_m, 'lr': learning_rate * 1}], weight_decay=0.1)

    scheduler = StepLR(optimizer, step_size=10, gamma=.8)

    best_accuracy = 0
    class_correct = [0] * output_dim  # 初始化每个类别的正确预测数
    class_total = [0] * output_dim  # 初始化每个类别的总样本数
    for epoch in range(num_epochs):
        train_correct = 0
        train_total = 0
        for inputs, labels in tqdm(train_loader):
            labels = labels.long().squeeze().to(device)
            optimizer.zero_grad()
            outputs, h, _ = model(inputs.cuda())
            loss = criterion(outputs, labels)
            loss.backward()

            # Updating parameters
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted.cpu() == labels.long().cpu()).sum()
        train_acc = 100. * train_correct.numpy() / train_total
        print('epoch: ', epoch, '. Loss: ', loss.item(), '. Train Acc: ', train_acc)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            test_correct = 0
            test_total = 0
            for inputs, labels in tqdm(test_loader):
                outputs,_, _  = model(inputs.cuda())
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted.cpu() == labels.long().squeeze().cpu()).sum()

                # 计算每个类别的正确预测
                for true, pred in zip(labels, predicted):
                    target = true.item()
                    class_correct[target] += (pred == target).item()
                    class_total[target] += 1

            test_acc = 100. * test_correct.numpy() / test_total
            if test_acc >= best_accuracy:
                torch.save(model.state_dict(), f'/data/mosttfzhu/MultiBarrelModel/github/Ev-Containers_MultiModel.pth')
                best_accuracy = test_acc
        print('epoch: ', epoch, '. Test Acc: ', test_acc, '. Best Acc: ', best_accuracy)

    return best_accuracy

if __name__ == '__main__':
    acc = train(init_b=0.46028508, init_tau_a=9.90546331, init_tau_m=4.34378016, batch_size=600)
    print(acc)
