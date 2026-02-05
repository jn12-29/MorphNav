import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d

def generate_spatial_ratemaps(x, y, hidden_states, n_bins=32, environment_bounds=None):
    """
    根据论文描述生成空间活动图 (Spatial Ratemaps)。
    
    参数:
        x (np.array): 形状为 (T,) 的横坐标轨迹。
        y (np.array): 形状为 (T,) 的纵坐标轨迹。
        hidden_states (np.array): 形状为 (T, N) 的神经元激活值，N为神经元数量。
        n_bins (int): 网格大小，论文中为 32。
        environment_bounds (tuple): 环境边界 (min_x, max_x, min_y, max_y)。
                                    如果不提供，将根据数据极值自动确定。
    
    返回:
        ratemaps (np.array): 形状为 (N, n_bins, n_bins) 的速率图。
                             未访问的区域填充为 NaN。
    """
    
    # 1. 确定环境边界
    if environment_bounds is None:
        # 如果未指定，使用数据的最小值和最大值（建议手动指定以保持不同episode一致）
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        # 稍微向外扩展一点点，防止边界点溢出
        epsilon = 1e-5
        range_x = [x_min - epsilon, x_max + epsilon]
        range_y = [y_min - epsilon, y_max + epsilon]
    else:
        range_x = [environment_bounds[0], environment_bounds[1]]
        range_y = [environment_bounds[2], environment_bounds[3]]

    num_neurons = hidden_states.shape[1]
    ratemaps = []

    # 2. 计算位置的占据图 (Occupancy Map) - 分母
    # 统计每个bin被访问了多少次
    occupancy, _, _, _ = binned_statistic_2d(
        x, y, 
        values=None, 
        statistic='count', 
        bins=n_bins, 
        range=[range_x, range_y]
    )

    # 3. 计算每个神经元的活动图 - 分子并求均值
    print(f"正在计算 {num_neurons} 个神经元的活动图...")
    
    for i in range(num_neurons):
        # 获取当前神经元的激活序列
        activations = hidden_states[:, i]
        
        # 统计该神经元在每个bin中的激活值总和
        act_sum, _, _, _ = binned_statistic_2d(
            x, y, 
            values=activations, 
            statistic='sum', 
            bins=n_bins, 
            range=[range_x, range_y]
        )
        
        # 4. 计算平均活动 (Mean Activity)
        # Mean = Sum / Count
        # 使用 np.divide 处理除以0的情况，分母为0的位置设为 NaN (对应论文中的 "unvisited bins")
        with np.errstate(divide='ignore', invalid='ignore'):
            rate_map = np.divide(act_sum, occupancy)
        
        # 论文提到 "without additional smoothing"，所以这里直接保存结果
        ratemaps.append(rate_map)

    return np.array(ratemaps)

# ==========================================
# 示例用法 (Mock Data)
# ==========================================

# 假设 T=10000 步，N=4 个神经元
T = 10000
N = 4

# 生成模拟轨迹 (随机游走)
t = np.linspace(0, 100, T)
true_x = np.sin(t) + np.random.normal(0, 0.1, T)
true_y = np.cos(t) + np.random.normal(0, 0.1, T)

# 生成模拟激活值 (假设神经元1喜欢在右上角发放)
# 这里仅为演示，实际 hidden_states 来自你的网络输出
activations = np.random.rand(T, N)
# 让神经元0在 (x>0, y>0) 区域更活跃
mask = (true_x > 0) & (true_y > 0)
activations[mask, 0] += 2.0 

# 运行函数
# 假设环境范围是 -1.5 到 1.5
maps = generate_spatial_ratemaps(
    true_x, true_y, activations, 
    n_bins=32, 
    environment_bounds=(-1.5, 1.5, -1.5, 1.5)
)

# 可视化结果
fig, axes = plt.subplots(1, N, figsize=(16, 4))
for i in range(N):
    # 使用 'nearest' 插值以展示原始像素块，符合 "without smoothing"
    im = axes[i].imshow(maps[i].T, origin='lower', cmap='jet', interpolation='nearest') 
    axes[i].set_title(f'Unit {i}')
    plt.colorbar(im, ax=axes[i])

plt.tight_layout()
plt.show()