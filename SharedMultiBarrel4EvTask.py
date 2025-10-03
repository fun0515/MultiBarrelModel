import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from torch.utils import data
import torch.nn.functional as F
from dataset import tactileDataset
from MultiBarrel4EvTask import TemporalCompressor, Readout, NeighborAggregator
from utils import mem_update_adp, SharedConv1dLinear
class SharedMultiBarrelModel(nn.Module):
    def __init__(self, num_barrels=39, input_size=2, output_size=36, init_b=0.02, init_tau_a = 5, init_tau_m = 10):
        super(SharedMultiBarrelModel, self).__init__()
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
                self.ThtoPops.append(SharedConv1dLinear(input_size=64, output_size=self.popSize[i]))

        # 生成神经元群落之间连接
        self.toPops = nn.ModuleList()
        for i, size in enumerate(self.popSize):
            inner_list = nn.ModuleList()
            for j, conn_prob in enumerate(self.AlltoAll[:, i]):
                if conn_prob != 0.:  # 不排除群落内部连接
                    inner_list.append(
                        SharedConv1dLinear(input_size=self.popSize[j], output_size=size))
            self.toPops.append(inner_list)

        # 生成动力学参数
        self.tau_adp, self.tau_m = nn.ParameterList(), nn.ParameterList()
        for size in self.popSize:
            self.tau_adp.append(nn.Parameter(torch.Tensor(size).cuda(), requires_grad=True))
            self.tau_m.append(nn.Parameter(torch.Tensor(size).cuda(), requires_grad=True))

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
        inputs = self.compressor(inputs) # (B,N,2,250) -> (B,N,64,64)
        batch_size, sensor_num, channel, seq_num = inputs.size()

        mems = [torch.rand(batch_size, self.num_barrels, size).cuda() for size in self.popSize]
        spikes = [torch.rand(batch_size, self.num_barrels, size).cuda() for size in self.popSize]
        self.b = [(torch.ones((self.num_barrels, 1)) * self.init_b).cuda() for _ in range(len(self.popSize))]

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
                        # torch.cat((input, spikes[6], spikes[7]), dim=-1))

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

    model = SharedMultiBarrelModel(num_barrels=39, output_size=output_dim, init_b=init_b, init_tau_a=init_tau_a, init_tau_m=init_tau_m)
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
                torch.save(model.state_dict(), f'/data/mosttfzhu/MultiBarrelModel/github/Ev-Containers_SharedMultiModel.pth')
                best_accuracy = test_acc
        print('epoch: ', epoch, '. Test Acc: ', test_acc, '. Best Acc: ', best_accuracy)

    return best_accuracy

if __name__ == '__main__':
    acc = train(init_b=0.46028508, init_tau_a=9.90546331, init_tau_m=4.34378016, batch_size=600)
    print(acc)
