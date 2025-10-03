'''
统计训练后多桶模型中桶间相关性，包括有无桶间电流
'''
import torch
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.pyplot as plt
from Model_lossLandscape import evaluate_model
from scipy.spatial.distance import cdist
from SharedMultiBarrel4EvTask import SharedMultiBarrelModel

def plot_correlation(model, dataset):
    """计算全时间长度桶活动相关系数"""
    _, h, _, _ = evaluate_model(model, dataset=dataset)  # (B, T, N_barrels, N_neurons)
    h_states = h.cpu().numpy()  # Sample_ID
    B, T, N_barrels, N_neurons = h_states.shape
    barrel_activity = h_states.mean(-1)  # (B, T, N_barrels)

    corr_mats = []
    for b in range(B):
        corr = np.corrcoef(barrel_activity[0].T)  # [39,39]
        corr_mats.append(corr)
    corr_mats = np.array(corr_mats) # (B,39,39)
    print(f"Mean corr: {np.mean(np.abs(corr_mats))}, std: {np.std(np.abs(corr_mats))}")

    barrel_activity_norm = (barrel_activity[0] - barrel_activity[0].min()) / (barrel_activity[0].max() - barrel_activity[0].min())
    plot_barrels_activity_lines(barrel_activity_norm)

def plot_barrels_activity_lines(barrel_activity):
    """绘制各个桶活动的时间序列堆叠图"""
    fig,ax = plt.subplots(figsize=(9, 5))
    for i in range(barrel_activity.shape[1]):
        ax.plot(barrel_activity[:, i] + i * 0.5, lw=0.8, alpha=1.)  # 纵向偏移避免重叠
    ax.set_xlabel('Time step')
    ax.set_ylabel('Barrels')

    ax.set_xlim(0,72)
    ax.set_ylim(0,19.5)
    ax.set_yticks([])

    # 调整边框粗细
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)  # 单位是磅（pt）

    plt.tight_layout()
    plt.show()

def plot_window_correlation(model, dataset, window_size=10, step=2):
    '''沿时间窗口计算局部桶活动相关系数'''
    _, h, _, _ = evaluate_model(model, dataset=dataset)  # (B, T, N_barrels, N_neurons)

    # 转换为NumPy数组并聚合神经元活动
    hidden_states = h.cpu().numpy()  # 形状 [B, T, N_barrels, N_neurons]
    B, T, N_barrels, _ = hidden_states.shape
    barrel_activity = hidden_states.sum(axis=-1)  # 形状 [B, T, N_barrels]

    # 计算滑动窗口索引
    max_start = T - window_size
    starts = np.arange(0, max_start + 1, step)
    window_indices = sliding_window_view(np.arange(T)[:starts[-1] + window_size], window_size)[::step]

    # 向量化计算相关系数矩阵
    corr_mats = []
    for b in range(B):
        # 生成滑动窗口
        windows = barrel_activity[b, window_indices]


        # 计算每个窗口的相关系数矩阵 [n_windows, N_barrels, N_barrels]
        batch_corr = np.array([np.corrcoef(window, rowvar=False) for window in windows])
        corr_mats.append(batch_corr)

    corr_mats = np.stack(corr_mats) # [B, n_windows, N_barrels, N_barrels]
    print('平均时间窗口的桶对相关性为：', np.mean(np.abs(corr_mats)),'标准差为:', np.nanstd(np.abs(corr_mats)))
    cal_window_pairs_topK(corr_mats)

def cal_window_pairs_topK(corr_mats, top_k=3):
    """统计各时间窗口的Top-K强相关桶对的邻居比例以及路径长度, 并可视化前五桶, 支持多样本输入 (形状 [B,T,N_barrels])"""
    coords = np.array([[-6, 0], [-5.3, -3], [-5.3, 3], [-4.6, -7.8], [-4.6, 7.8],
                       [-3.5, 0], [-3.05, -5.2], [-3.05, 5.2], [-3.1, -1.75], [-3.1, 1.75],
                       [-1.6, -8.9], [-1.75, -3], [-1.6, 8.9], [-1.75, 3], [-1.5, 0],
                       [-0.7, -1.3], [-0.7, 1.3], [0, -6], [0, 6], [0, -3.5],
                       [0, 0], [0, 3.5], [0.8, -1.3], [0.8, 1.3], [1.6, -8.9],
                       [1.6, 8.9], [1.5, 0], [1.75, 3], [1.75, -3], [3.05, -5.2],
                       [3.1, 1.75], [3.05, 5.2], [3.1, -1.75], [3.6, 0], [4.6, -7.8],
                       [4.6, 7.8], [5.3, -3], [5.3, 3], [6, 0]])
    B, N_win, N, _ = corr_mats.shape
    dist_matrix = cdist(coords, coords)

    # 收集所有窗口的Top-K桶对
    all_pairs = []

    for b in range(B):
        for w in range(N_win):
            corr = corr_mats[b, w]
            triu_i, triu_j = np.triu_indices_from(corr, k=1)
            corr_flat = corr[triu_i, triu_j]

            # 按绝对值取当前窗口的Top-K
            abs_corr = np.abs(corr_flat)
            topk_idx = np.argsort(abs_corr)[-top_k:][::-1]  # 降序取前top_k

            # 记录候选对（允许重复）
            for idx in topk_idx:
                i, j = triu_i[idx], triu_j[idx]
                all_pairs.append((i, j, abs_corr[idx], dist_matrix[i, j])) # (i, j, 相关性，距离)

    # 统计频次与平均强度
    from collections import defaultdict
    pair_stats = defaultdict(lambda: {'count': 0, 'total_strength': 0.0, 'distance': 0.0})

    for i, j, strength, dist in all_pairs:
        key = tuple(sorted((i, j)))  # 保证(i,j)和(j,i)视为同一对
        pair_stats[key]['count'] += 1
        pair_stats[key]['total_strength'] += strength
        pair_stats[key]['distance'] = dist_matrix[i, j]  # 距离固定，取首次出现的值

    # 按频次排序（频次相同则按强度排序）
    sorted_pairs = sorted(pair_stats.items(),
                          key=lambda x: (-x[1]['count'], -x[1]['total_strength']))[:5]

    # 格式转换
    top5_pairs = []
    top5_avg_strength = []
    top5_distances = []

    for (pair, stats) in sorted_pairs:
        i, j = pair
        top5_pairs.append([i, j])
        top5_avg_strength.append(stats['total_strength'] / stats['count'])
        top5_distances.append(stats['distance'])

    all_pairs = np.array(all_pairs)
    print('全局平均强相关距离:', np.nanmean(all_pairs[:, 3]))

    # 输出统计结果
    print("前五强相关性:", top5_avg_strength)
    print("前五强距离:", top5_distances)
    print("前五强平均距离:", np.mean(top5_distances))

    # 绘制前五强桶对
    fig, ax = plt.subplots(figsize=(6, 8))

    # 绘制所有桶（灰色）
    ax.scatter(
        coords[:, 0], coords[:, 1],
        c='lightgrey',  # 统一灰色
        s=300,  # 保持原大小
        edgecolor='black',  # 黑色边框
        linewidth=1,
        alpha=0.8,
        zorder=1  # 底层绘制
    )

    # 绘制Top5连接线
    for (i, j) in top5_pairs:
        x_pair = [coords[i, 0], coords[j, 0]]
        y_pair = [coords[i, 1], coords[j, 1]]
        ax.plot(x_pair, y_pair,
                color='#E74C3C',  # 红色突出显示
                linewidth=3,
                alpha=0.8,
                solid_capstyle='round',
                zorder=2  # 顶层绘制
                )

    # 坐标设置
    ax.tick_params(axis='both', labelbottom=False, labelleft=False)
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=1.5)

    # 调整边框粗细
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    # 保持纵横比
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()



if __name__ == '__main__':
    model = SharedMultiBarrelModel(output_size=20, init_b=0.46028508, init_tau_a=9.90546331, init_tau_m=4.34378016).cuda()
    model.load_state_dict(torch.load(f'/data/mosttfzhu/MultiBarrelModel/github/Ev-Containers_SharedMultiModel.pth'))

    plot_correlation(model, dataset='C')
    plot_window_correlation(model, dataset='C')