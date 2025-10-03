'''
绘制共享参数多桶模型与独立参数多桶模型的损失曲面图
'''
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
from SharedMultiBarrel4EvTask import SharedMultiBarrelModel
from MultiBarrel4EvTask import MultiBarrelModel, TemporalCompressor, NeighborAggregator, Readout
from dataset import tactileDataset

plt.rcParams['font.family'] = ['Times New Roman', 'serif']
plt.rcParams['font.size'] = 26

def evaluate_model(model, dataset=None):
    if dataset == 'O':
        test_dataset = tactileDataset(r'./data/Ev-Objects', train=False)
        test_loader = DataLoader(test_dataset, batch_size=600, shuffle=False)
        data, labels = next(iter(test_loader))
    elif dataset == 'C':
        test_dataset = tactileDataset(r'./data/Ev-Containers', train=False)
        test_loader = DataLoader(test_dataset, batch_size=600, shuffle=False)
        data, labels = next(iter(test_loader))
    model.eval()
    with torch.no_grad():
        outputs, h, _ = model(data.cuda())
        _, predicted = torch.max(outputs.data, 1)
        test_total = labels.size(0)
        test_correct = (predicted.cpu() == labels.long().squeeze().cpu()).sum()
        test_acc = 100. * test_correct.numpy() / test_total
    return test_acc, h, data, labels

def extract_barrel_params(model):
    """提取所有桶相关参数并展平为向量"""
    params = []
    # 提取动力学参数: tau_adp 和 tau_m
    for param_list in [model.tau_adp, model.tau_m]:
        for tensor in param_list:
            params.append(tensor.data.flatten())

    # 提取ThtoPops参数（丘脑到群落的连接）
    for module in model.ThtoPops:
        params.append(module.conv.weight.data.flatten())  # 卷积权重
        if module.conv.bias is not None:
            params.append(module.conv.bias.data.flatten())  # 偏置

    # 提取toPops参数（群落间连接）
    for pop_module_list in model.toPops:
        for module in pop_module_list:
            params.append(module.conv.weight.data.flatten())  # 卷积权重
            if module.conv.bias is not None:
                params.append(module.conv.bias.data.flatten())  # 偏置

    return torch.cat(params)  # 返回展平后的向量 (D, )


def apply_perturbation(model, perturbation):
    """将扰动后的参数加载回模型"""
    pointer = 0

    # 1. 重新填充动力学参数 Tau_m 和 Tau_adp
    for param_list in [model.tau_m, model.tau_adp]:
        for tensor in param_list:
            numel = tensor.numel()
            tensor.data.copy_(perturbation[pointer:pointer+numel].view_as(tensor))
            pointer += numel

    # 2. 重新填充ThtoPops参数
    for module in model.ThtoPops:
        numel_weight = module.conv.weight.numel()
        module.conv.weight.data.copy_(
            perturbation[pointer:pointer + numel_weight].view_as(module.conv.weight)
        )
        pointer += numel_weight
        if hasattr(module.conv, 'bias') and module.conv.bias is not None:
            numel_bias = module.conv.bias.numel()
            module.conv.bias.data.copy_(
                perturbation[pointer:pointer + numel_bias].view_as(module.conv.bias)
            )
            pointer += numel_bias

    # 3. 重新填充toPops参数
    for pop_module_list in model.toPops:
        for module in pop_module_list:
            numel_weight = module.conv.weight.numel()
            module.conv.weight.data.copy_(
                perturbation[pointer:pointer + numel_weight].view_as(module.conv.weight)
            )
            pointer += numel_weight
            if hasattr(module.conv, 'bias') and module.conv.bias is not None:
                numel_bias = module.conv.bias.numel()
                module.conv.bias.data.copy_(
                    perturbation[pointer:pointer + numel_bias].view_as(module.conv.bias)
                )
                pointer += numel_bias


def gram_schmidt(v1, v2):
    """Gram-Schmidt正交化确保扰动方向正交"""
    v2 = v2 - (v1 @ v2) * v1 / (v1.norm() ** 2 + 1e-8)
    return v1, v2 / (v2.norm() + 1e-8)


def plot_3d_loss_landscape(model, dataloader, perturbation_scale=1., grid_size=3):
    # 1. 加载预训练模型
    model.eval()

    # 2. 提取桶参数并备份
    original_params = extract_barrel_params(model)  # (D,)
    D = original_params.shape[0]
    print(f"Total barrel parameters dimension: {D}")

    # 3. 生成两个随机正交扰动方向
    d1 = torch.randn_like(original_params)  # 随机方向1
    d2 = torch.randn_like(original_params)  # 随机方向2
    d1, d2 = gram_schmidt(d1, d2)  # 正交化
    d1 = d1 * perturbation_scale / d1.norm()  # 标准化扰动强度
    d2 = d2 * perturbation_scale / d2.norm()

    # 4. 生成扰动网格
    alpha_range = np.linspace(-10, 10, grid_size)  # 方向1的扰动系数
    beta_range = np.linspace(-10, 10, grid_size)  # 方向2的扰动系数
    loss_grid = np.zeros((grid_size, grid_size))

    criterion = nn.CrossEntropyLoss()
    # 5. 遍历网格点计算损失
    original_backup = original_params.clone()  # 备份原始参数
    for i, alpha in enumerate(alpha_range):
        for j, beta in enumerate(beta_range):
            # 施加扰动
            perturbed = original_backup + alpha * d1 + beta * d2
            apply_perturbation(model, perturbed)

            test_acc, _, _, _ = evaluate_model(model, dataset='C')
            print(test_acc)

            # 计算验证损失（固定数据批次确保公平）
            with torch.no_grad():
                for inputs, targets in dataloader:
                    inputs, targets = inputs.cuda(), targets.long().squeeze().cuda()
                    outputs, _, _ = model(inputs)
                    loss = criterion(outputs, targets)
                print(loss)
                loss_grid[i, j] = loss
            # 恢复原始参数
            apply_perturbation(model, original_backup)

    # 6. 绘制3D曲面
    X, Y = np.meshgrid(alpha_range, beta_range)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, loss_grid.T, cmap='viridis',
                           rstride=1, cstride=1, alpha=0.8)
    ax.contour(X, Y, loss_grid.T, zdir='z', offset=loss_grid.min(), cmap='viridis')
    ax.set_xlabel(r'$\alpha$', fontsize=30, labelpad=15)
    ax.set_ylabel(r'$\beta$', fontsize=30, labelpad=15)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    model = MultiBarrelModel(num_barrels=39, output_size=20, init_b=0.46028508, init_tau_a=9.90546331, init_tau_m=4.34378016).cuda()
    model.load_state_dict(torch.load(f'/data/mosttfzhu/MultiBarrelModel/github/Ev-Containers_MultiModel.pth'))

    Objects_dataset = tactileDataset(r'./data/Ev-Objects', train=True)
    Objects_loader = DataLoader(Objects_dataset, batch_size=600, shuffle=False)
    Containers_dataset = tactileDataset(r'./data/Ev-Containers', train=True)
    Containers_loader = DataLoader(Containers_dataset, batch_size=600, shuffle=False)

    plot_3d_loss_landscape(model, dataloader=Containers_loader, perturbation_scale = 2., grid_size=15)