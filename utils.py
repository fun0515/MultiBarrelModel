'''
ALIF单神经元动力学、神经元共享连接等辅助模块
'''
import torch
import torch.nn as nn
import math

lens = 0.5  # hyper-parameters of approximate function
b_j0 = 0.1
R_m = 1  # membrane resistance
dt = 1  #
gamma = .5  # gradient scale

class ActFun_adp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()

        temp = torch.exp(-(input**2)/(2*lens**2))/torch.sqrt(2*torch.tensor(math.pi))/lens
        return grad_input * temp.float() * gamma


def mem_update_adp(inputs, mem, spike, tau_adp, b, tau_m, dt=1, isAdapt=1):
    alpha = torch.exp(-1. * dt / tau_m).cuda()
    ro = torch.exp(-1. * dt / tau_adp).cuda()
    if isAdapt:
        beta = 1.8
    else:
        beta = 0.

    b = ro * b + (1 - ro) * spike
    B = b_j0 + beta * b

    mem = mem * alpha + (1 - alpha) * R_m * (inputs + torch.normal(mean=0,std=1.,size=(inputs.size())).to('cuda')) # 更新当前时刻膜电位, (B, num_barrels, size)

    # 判断是否发放脉冲
    inputs_ = mem - B
    spike = act_fun_adp(inputs_)

    mem = mem - B * spike * dt # 发放脉冲神经元的膜电位复位
    return mem, spike, B, b

# 没有脉冲诱导的阈值增加
def mem_update(inputs, mem, tau_m, B, dt=1):
    alpha = torch.exp(-1. * dt / tau_m).cuda()
    mem = mem * alpha + (1 - alpha) * R_m * inputs

    # 判断是否发放脉冲
    inputs_ = mem - B
    spike = act_fun_adp(inputs_)

    mem = mem - B * spike # 发放脉冲神经元的膜电位复位
    return mem, spike, B,

act_fun_adp = ActFun_adp.apply


class Conv1dLinear(nn.Module):
    def __init__(self, num_barrels, input_size, output_size):
        super().__init__()
        # 每个桶独立权重
        self.conv = nn.Conv1d(
            in_channels=input_size * num_barrels,
            out_channels=output_size * num_barrels,
            kernel_size=1,
            groups=num_barrels  # 分组卷积，独立权重
        )
        self.num_barrels = num_barrels
        self.bn = nn.BatchNorm1d(output_size * num_barrels)

    def forward(self, x):
        # 输入形状: (B, num_barrels, input_size)
        B, N, D = x.shape
        x = x.view(B, N * D, 1)  # (B, N*D, 1)
        out = self.conv(x)       # (B, N*O, 1)
        out = self.bn(out)
        return out.view(B, N, -1)  # (B, N, O)

class SharedConv1dLinear(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        # 所有桶共享的卷积核
        self.conv = nn.Conv1d(input_size, output_size, kernel_size=1)
        self.bn = nn.BatchNorm1d(output_size)

    def forward(self, x):
        """输入x: (B, N=39, D_in)"""
        B, N, D = x.size()
        # 合并批次和桶维度
        x = x.view(B * N, D, 1)  # (B*39, D, 1)
        x = self.conv(x)  # (B*39, D_out, 1)
        x = self.bn(x)  # 共享BN参数
        return x.view(B, N, -1)  # (B, 39, D_out)





