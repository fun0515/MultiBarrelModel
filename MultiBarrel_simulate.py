'''
可视化初始多桶模型中桶活动的扩散传播现象；
仿照光遗传实验，用恒定电流只刺激中央桶；
'''
import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from MultiBarrel4EvTask import TemporalCompressor, Readout, NeighborAggregator
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from utils import Conv1dLinear

def set_seed(seed):
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # NumPy
    np.random.seed(seed)

set_seed(515)
b_j0 = 0.46028508
class ActFun_adp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):  # input = membrane potential- threshold
        ctx.save_for_backward(input)
        return input.gt(0).float()  # is firing ???

    @staticmethod
    def backward(ctx, grad_output):  # approximate the gradients
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()

        temp = torch.exp(-(input**2)/(2*0.5**2))/torch.sqrt(2*torch.tensor(math.pi))/0.5
        return grad_input * temp.float() * 0.5
def mem_update_adp(inputs, mem, spike, tau_adp, b, tau_m, R_m=1, dt=1, isAdapt=1):
    alpha = torch.exp(-1. * dt / tau_m).cuda()
    ro = torch.exp(-1. * dt / tau_adp).cuda()
    if isAdapt:
        beta = 1.8
    else:
        beta = 0.

    b = ro * b + (1 - ro) * spike
    B = b_j0 + beta * b

    mem = mem * alpha + (1 - alpha) * R_m * inputs #+ torch.normal(mean=0,std=1.,size=(inputs.size())).to('cuda')) # 更新当前时刻膜电位, (B, num_barrels, size)

    # 判断是否发放脉冲
    inputs_ = mem - B
    spike = act_fun_adp(inputs_)  # act_fun : approximation firing function

    mem = mem - B * spike * dt # 发放脉冲神经元的膜电位复位
    return mem, spike, B, b
act_fun_adp = ActFun_adp.apply
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
        self.neighbor_agg = NeighborAggregator(k=3, select_ratio=0.1)

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
        batch_size, sensor_num, channel, seq_num = inputs.size()  #

        mems = [torch.zeros(batch_size, self.num_barrels, size).cuda() for size in self.popSize]
        spikes = [torch.zeros(batch_size, self.num_barrels, size).cuda() for size in self.popSize]
        self.b = [(torch.zeros((self.num_barrels, 1)) * self.init_b).cuda() for _ in range(len(self.popSize))]

        L5_state, h_state, m_state = [], [], []
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
                if t >0:
                    input_current += self.neighbor_agg(x=input,spikes=torch.cat(spikes,dim=-1))

                # 更新神经元状态
                mems[i], spikes[i], B, self.b[i] = mem_update_adp(input_current, mems[i], spikes[i], self.tau_adp[i],
                                                                  self.b[i], self.tau_m[i])

                pre_mem_state[i] = mems[i] + spikes[i] * B
            h_state.append(torch.cat(spikes,dim=-1)) # (B, N_barrels, N_neurons)
            L5_state.append(torch.cat((spikes[4],spikes[5], spikes[6],spikes[7]),dim=-1)) # (B,39,2*popsize)
            m_state.append(torch.cat((pre_mem_state), dim=-1))
        h_state = torch.stack(h_state,dim=1) # (B, T, N_barrels, N_neurons)
        m_state = torch.stack(m_state, dim=1)  # (B, T, N_barrels, N_neurons)
        L5_state = torch.stack(L5_state, dim=1)  # (B,T,39,2*popsize)
        output = self.readout(L5_state)
        return F.softmax(output, dim=1), h_state, m_state

def simulate_initModel():
    init_b, init_tau_a, init_tau_m = 0.46028508, 9.90546331, 4.34378016
    input_dim, output_dim = 2, 36

    inputs = torch.zeros(size=(2,39,2,250))
    inputs[0,20,:,:] = 1.

    model = MultiBarrelModel(input_size=input_dim, output_size=output_dim, init_b=init_b, init_tau_a=init_tau_a,
                              init_tau_m=init_tau_m).cuda()
    with torch.no_grad():
        _, h, m = model(inputs.cuda())
    # 绘制1-2时刻的桶神经活动
    plot_barrels_activity(h_states=h, time_range=(1,2))

def plot_barrels_activity(h_states, time_range=(0, 1), cmap='viridis', annotate=False):
    """
    绘制基于传感器坐标的放电率热力图
    """
    sample_idx = 0
    # 数据预处理
    coords = np.array([[-6, 0], [-5.3, -3], [-5.3, 3], [-4.6, -7.8], [-4.6, 7.8],
                       [-3.5, 0], [-3.05, -5.2], [-3.05, 5.2], [-3.1, -1.75], [-3.1, 1.75],  # 10
                       [-1.6, -8.9], [-1.75, -3], [-1.6, 8.9], [-1.75, 3], [-1.5, 0],
                       [-0.7, -1.3], [-0.7, 1.3], [0, -6], [0, 6], [0, -3.5],
                       [0, 0], [0, 3.5], [0.8, -1.3], [0.8, 1.3], [1.6, -8.9],
                       [1.6, 8.9], [1.5, 0], [1.75, 3], [1.75, -3], [3.05, -5.2],
                       [3.1, 1.75], [3.05, 5.2], [3.1, -1.75], [3.6, 0], [4.6, -7.8],
                       [4.6, 7.8], [5.3, -3], [5.3, 3], [6, 0]
                       ])
    activity = h_states[sample_idx, time_range[0]:time_range[1] + 1, :, :]
    # 计算放电率特征 (T,N_barrels,N_neurons) -> (N_barrels,)
    rates = activity.sum(dim=(0, 2)).cpu().numpy() / activity.size(0)  # 时间求和 + 神经元求和
    norm = Normalize(vmin=0, vmax=40)

    # 创建画布
    fig, ax = plt.subplots(figsize=(6, 8))

    # 绘制散点
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=rates,
        s=300, #+ 500 * (rates - rates.min()) / (rates.max() - rates.min()),  # 大小映射
        cmap=cmap,
        edgecolor='black',
        linewidth=1,
        alpha=0.8,
        norm=norm
    )

    # 标注传感器编号
    if annotate:
        for i, (x, y) in enumerate(coords):
            plt.text(x, y, f"{i + 1}\n{rates[i]:.1f}",
                     ha='center', va='center',
                     fontsize=8, color='white' if rates[i] > 0.7 * rates.max() else 'black')

    # 添加颜色条
    cbar = plt.colorbar(scatter, shrink=0.8)
    cbar.ax.tick_params(labelsize=25)
    cbar.set_label('Firing Rate')

    # 坐标设置
    plt.title(f"Timestep {time_range[0]}-{time_range[1]}")
    plt.xlabel("X Coordinate", fontsize=25)
    plt.ylabel("Y Coordinate", fontsize=25)
    ax.tick_params(axis='both', labelbottom=False, labelleft=False)
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=1.5)

    # 调整边框粗细
    for spine in ax.spines.values():
        spine.set_linewidth(2)  # 单位是磅（pt）

    # 保持纵横比
    plt.tight_layout()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

if __name__ == '__main__':
    simulate_initModel()