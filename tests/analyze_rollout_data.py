"""
Spatial Analysis Script for Rollout Data
========================================
使用 DataRecorder 加载 rollout data 中的数据，生成空间活动图。

参考: generate_spatiai_ratemaps.py 中的 spatial ratemap 生成方法
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
from pathlib import Path
import sys
from tqdm import tqdm
from scipy.ndimage import rotate
import warnings
from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter1d
import os

sys.path.append("/home/xh/ai4neuron/MorphNav/rl-baselines3-zoo")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import components
from rl_zoo3.data_recorder import DataRecorder


def extract_position_from_qpos(qpos):
    """
    从 qpos 中提取 (x, y) 位置。
    MuJoCo 的 qpos 通常是 [qx, qy, qz, ...] 或 [x, y, z, ...] 格式。
    对于 2D 平面任务，通常是 [x, y, ...]。
    """
    if qpos is None or len(qpos) == 0:
        return None, None

    if len(qpos) >= 2:
        x = qpos[:, 0] if qpos.ndim == 2 else np.array([q[0] for q in qpos])
        y = qpos[:, 1] if qpos.ndim == 2 else np.array([q[1] for q in qpos])
        return x, y

    return None, None


def flatten_activations(activations_dict):
    """
    将 activations 字典展平为 2D 数组 (T, N)。
    处理不同层级的激活值。
    """
    # import pdb

    # pdb.set_trace()
    all_activations = []
    layer_names = []

    for layer_name, values in activations_dict.items():
        arr = np.array(values)
        if arr.size == arr.shape[0] * arr.shape[-1]:
            # from (T, 1,.., D) to (T, D)
            arr = arr.reshape(arr.shape[0], -1)
        else:
            raise ValueError(f"Layer {layer_name} has unexpected shape {arr.shape}")

        all_activations.append(arr)
        layer_names.append(layer_name)

    # stack [[T, D1], [T, D2], ...] to [T, D1+D2+...]
    stacked = np.hstack(all_activations)

    return stacked, layer_names


def generate_spatial_ratemaps(x, y, hidden_states, n_bins=32, environment_bounds=None):
    """
    根据论文描述生成空间活动图 (Spatial Ratemaps)。

    参数:
        x (np.array): 形状为 (T,) 的横坐标轨迹。
        y (np.array): 形状为 (T,) 的纵坐标轨迹。
        hidden_states (np.array): 形状为 (T, N) 的神经元激活值。
        n_bins (int): 网格大小，论文中为 32。
        environment_bounds (tuple): 环境边界 (min_x, max_x, min_y, max_y)。

    返回:
        ratemaps (np.array): 形状为 (N, n_bins, n_bins) 的速率图。
    """
    if environment_bounds is None:
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        epsilon = 1e-5
        range_x = [x_min - epsilon, x_max + epsilon]
        range_y = [y_min - epsilon, y_max + epsilon]
    else:
        range_x = [environment_bounds[0], environment_bounds[1]]
        range_y = [environment_bounds[2], environment_bounds[3]]

    num_neurons = hidden_states.shape[1]
    ratemaps = []

    occupancy, _, _, _ = binned_statistic_2d(
        x, y, values=None, statistic="count", bins=n_bins, range=[range_x, range_y]
    )

    print(f"正在计算 {num_neurons} 个神经元/维度的活动图...")

    for i in range(num_neurons):
        activations = hidden_states[:, i]

        act_sum, _, _, _ = binned_statistic_2d(
            x,
            y,
            values=activations,
            statistic="sum",
            bins=n_bins,
            range=[range_x, range_y],
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            rate_map = np.divide(act_sum, occupancy)

        ratemaps.append(rate_map)

    return np.array(ratemaps)


def generate_spatial_ratemaps_smooth(
    x, y, hidden_states, n_bins=32, environment_bounds=None, smooth_sigma=1.5
):
    """
    改进版：加入高斯平滑，更符合生物学网格细胞的分析标准。
    smooth_sigma: 高斯核的标准差 (单位为 bin 的数量)，通常设为 1.0 到 2.0。
    """
    if environment_bounds is None:
        epsilon = 1e-5
        range_x = [np.min(x) - epsilon, np.max(x) + epsilon]
        range_y = [np.min(y) - epsilon, np.max(y) + epsilon]
    else:
        range_x = [environment_bounds[0], environment_bounds[1]]
        range_y = [environment_bounds[2], environment_bounds[3]]

    num_neurons = hidden_states.shape[1]
    ratemaps = []

    # 计算访问次数 (Occupancy)
    occupancy, _, _, _ = binned_statistic_2d(
        x, y, values=None, statistic="count", bins=n_bins, range=[range_x, range_y]
    )

    print(f"正在计算并平滑 {num_neurons} 个神经元的活动图 (sigma={smooth_sigma})...")

    # 对 occupancy 进行平滑
    smoothed_occupancy = gaussian_filter(occupancy, sigma=smooth_sigma)

    for i in range(num_neurons):
        activations = hidden_states[:, i]

        # 计算激活总和 (Activation Sum)
        act_sum, _, _, _ = binned_statistic_2d(
            x,
            y,
            values=activations,
            statistic="sum",
            bins=n_bins,
            range=[range_x, range_y],
        )

        # 对激活总和进行平滑
        smoothed_act = gaussian_filter(act_sum, sigma=smooth_sigma)

        # 相除得到平滑后的速率图
        with np.errstate(divide="ignore", invalid="ignore"):
            rate_map = np.divide(smoothed_act, smoothed_occupancy)

        # 核心：将原本就没有访问过的区域（未平滑的 occupancy 为 0 的地方）重新置为 NaN
        # 这样平滑后的数据就不会“渗漏”到障碍物里
        rate_map[occupancy == 0] = np.nan

        ratemaps.append(rate_map)

    return np.array(ratemaps)


def resolve_output_dir(data_dir: str, output_dir: str | None = None) -> str:
    if output_dir:
        return output_dir
    return str(Path(data_dir).resolve().parent / "analysis")


def analyze_rollout_data(
    data_dir: str,
    output_dir: str | None = None,
    n_bins: int = 32,
    environment_bounds: tuple = None,
    max_episodes: int = None,
    plot_mode: str = "grid",
    max_display: int = None,
    dpi: int = 100,
):
    """
    分析 rollout data 中的所有 episode，生成空间活动图。

    参数:
        data_dir: 数据目录
        output_dir: 输出目录
        n_bins: 网格大小
        environment_bounds: 环境边界 (min_x, max_x, min_y, max_y)
        max_episodes: 最大处理的 episode 数量
        plot_mode: 'grid' (网格拼图), 'individual' (每个单元单独保存), 'sample' (随机采样)
        max_display: 最大显示数量 (用于 grid 和 sample 模式)
        dpi: 图像分辨率
    """
    output_dir = resolve_output_dir(data_dir, output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"加载数据 from {data_dir}...")

    episodes = DataRecorder.load_all_episodes(data_dir, prefix="episode")

    if not episodes:
        print(f"在 {data_dir} 中未找到 episode 数据！")
        return

    print(f"找到 {len(episodes)} 个 episodes")

    if max_episodes:
        episodes = episodes[:max_episodes]
        print(f"处理前 {max_episodes} 个 episodes")

    all_x, all_y = [], []
    combined_activations = None

    for idx, ep in enumerate(episodes):
        print(f"\n处理 Episode {idx}...")
        print(f"  - Episode 长度: {len(ep['actions'])}")

        qpos = ep.get("qpos")
        x, y = extract_position_from_qpos(qpos)

        if x is not None and y is not None:
            print(
                f"  - 位置范围: x=[{x.min():.3f}, {x.max():.3f}], y=[{y.min():.3f}, {y.max():.3f}]"
            )
            all_x.extend(x.tolist())
            all_y.extend(y.tolist())

        activations = ep.get("activations", {})
        if activations:
            combined_activations, layer_names = flatten_activations(activations)

    if not all_x:
        print("错误：未能提取位置数据！")
        return

    all_x = np.array(all_x)
    all_y = np.array(all_y)
    print(f"\n总计: {len(all_x)} 步")
    print(
        f"位置范围: x=[{all_x.min():.3f}, {all_x.max():.3f}], y=[{all_y.min():.3f}, {all_y.max():.3f}]"
    )

    if environment_bounds is None:
        environment_bounds = (
            all_x.min() - 0.1,
            all_x.max() + 0.1,
            all_y.min() - 0.1,
            all_y.max() + 0.1,
        )
        print(f"自动确定环境边界: {environment_bounds}")

    min_len = min(len(all_x), len(combined_activations))
    all_x = all_x[:min_len]
    all_y = all_y[:min_len]
    combined_activations = combined_activations[:min_len]

    ratemaps = generate_spatial_ratemaps_smooth(
        all_x,
        all_y,
        combined_activations,
        n_bins=n_bins,
        environment_bounds=environment_bounds,
    )

    print(f"\n生成 {ratemaps.shape[0]} 个空间活动图，形状: {ratemaps.shape[1:]}")
    # ================= 新增：方向细胞 (HD Cells) 分析 =================
    # 提取速度分量 (Unit 0 为 vx, Unit 1 为 vy)
    vx = combined_activations[:, 2]
    vy = combined_activations[:, 3]

    # 利用 np.arctan2 计算运动角度 (-pi 到 pi)
    angles = np.arctan2(vy, vx)

    # 计算调谐曲线和 MVL 得分
    tuning_curves, hd_mvls, hd_bin_centers = analyze_head_direction_cells(
        angles=angles,
        hidden_states=combined_activations,
        n_angle_bins=60,
        smooth_sigma=1.0,
    )

    # 找出方向得分(MVL)最高的神经元 (忽略纯静止/全 NaN 的情况)
    valid_hd_idx = np.where(~np.isnan(hd_mvls))[0]
    if len(valid_hd_idx) > 0:
        # 注意: Unit0 和 Unit1 作为输入，它们的 MVL 必定极高，如果不想看它们，可以在排序前过滤掉
        best_hd_unit = valid_hd_idx[np.argsort(hd_mvls[valid_hd_idx])[::-1][0]]
        print(
            f"\n最高方向得分(MVL)为: {hd_mvls[best_hd_unit]:.3f} (Unit {best_hd_unit})"
        )
    else:
        print("\n未找到有效的方向得分。")

    # 画出 Top K 的方向细胞
    plot_best_hd_cells(tuning_curves, hd_mvls, hd_bin_centers, output_dir, top_k=5)
    # ================= 新增：网格细胞分析 =================
    autocorrs, grid_scores = analyze_grid_cells(ratemaps)

    # 找出网格得分最高的神经元（过滤掉 NaN）
    valid_idx = np.where(~np.isnan(grid_scores))[0]
    if len(valid_idx) > 0:
        # 正确获取绝对索引
        best_unit = valid_idx[np.argsort(grid_scores[valid_idx])[::-1][0]]
        print(f"\n最高网格得分为: {grid_scores[best_unit]:.3f} (Unit {best_unit})")
    else:
        print("\n未找到有效的网格得分。")

    print("正在绘制最佳网格细胞对比图...")
    plot_best_grid_cells(
        ratemaps=ratemaps,
        autocorrs=autocorrs,
        grid_scores=grid_scores,
        output_dir=output_dir,
        top_k=5,  # 你可以根据需要调整这个数字，比如改成 10 查看前十名
    )

    # =====================================================

    plot_ratemaps(
        ratemaps,
        output_dir,
        combined_activations.shape[1],
        plot_mode=plot_mode,
        max_display=max_display,
        dpi=dpi,
    )

    plot_position_distribution(
        all_x, all_y, environment_bounds, n_bins, output_dir, dpi
    )

    # 在最终的保存步骤中，把 HD 细胞的数据也一并存入 npz：
    np.savez(
        f"{output_dir}/analysis_data.npz",
        x=all_x,
        y=all_y,
        angles=angles,  # 保存角度数据
        ratemaps=ratemaps if combined_activations is not None else None,
        autocorrs=autocorrs,
        grid_scores=grid_scores,
        hd_tuning_curves=tuning_curves,  # 保存方向调谐曲线
        hd_mvls=hd_mvls,  # 保存 MVL 得分
        hd_bin_centers=hd_bin_centers,
        bounds=environment_bounds,
    )
    print(f"\n分析结果已保存到 {output_dir}/")


def plot_ratemaps(
    ratemaps,
    output_dir,
    num_units,
    plot_mode="grid",
    max_display=None,
    dpi=100,
):
    """
    可视化空间活动图。

    参数:
        ratemaps: 形状为 (N, n_bins, n_bins) 的速率图数组
        output_dir: 输出目录
        num_units: 总单元数
        plot_mode: 'grid' (网格拼图), 'individual' (每个单元单独保存), 'sample' (随机采样)
        max_display: 最大显示数量 (用于 grid 和 sample 模式)
        dpi: 图像分辨率
    """
    n_units = ratemaps.shape[0]

    if plot_mode == "individual":
        _plot_individual(ratemaps, output_dir, dpi)
    elif plot_mode == "sample":
        _plot_sample(ratemaps, output_dir, num_units, max_display, dpi)
    else:
        _plot_grid(ratemaps, output_dir, num_units, max_display, dpi)


def _plot_grid(ratemaps, output_dir, num_units, max_display=None, dpi=100):
    """网格拼图模式：所有单元拼成一个大图。"""
    n_units = ratemaps.shape[0]
    max_display = min(max_display or min(100, n_units), n_units)

    if max_display >= 100:
        n_cols = 10
    elif max_display >= 64:
        n_cols = 8
    elif max_display >= 36:
        n_cols = 6
    else:
        n_cols = min(8, max_display)

    n_rows = int(np.ceil(max_display / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(2.5 * n_cols, 2.5 * n_rows), squeeze=False
    )
    axes = axes.flatten()

    for i in range(max_display):
        ratemap = ratemaps[i]
        vmin = np.nanmin(ratemap)
        vmax = np.nanmax(ratemap)

        if vmin == vmax:
            im = axes[i].imshow(
                ratemap.T, origin="lower", cmap="gray", interpolation="nearest"
            )
        else:
            im = axes[i].imshow(
                ratemap.T,
                origin="lower",
                cmap="jet",
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
        axes[i].set_title(f"Unit {i}", fontsize=8)
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    for i in range(max_display, len(axes)):
        axes[i].axis("off")

    plt.suptitle(
        f"Spatial Ratemaps (showing {max_display}/{num_units} units, adaptive scale)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/spatial_ratemaps_grid.png", dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"网格图已保存到 {output_dir}/spatial_ratemaps_grid.png")


def _plot_individual(ratemaps, output_dir, dpi=100):
    """单独保存模式：每个单元保存为单独的图像文件。"""
    n_units = ratemaps.shape[0]
    individual_dir = Path(output_dir) / "individual_ratemaps"
    individual_dir.mkdir(parents=True, exist_ok=True)

    print(f"正在保存 {n_units} 张单独的图片到 {individual_dir}/")

    for i in range(n_units):
        ratemap = ratemaps[i]
        vmin = np.nanmin(ratemap)
        vmax = np.nanmax(ratemap)

        fig, ax = plt.subplots(figsize=(4, 3.5))

        if vmin == vmax:
            im = ax.imshow(
                ratemap.T, origin="lower", cmap="gray", interpolation="nearest"
            )
        else:
            im = ax.imshow(
                ratemap.T,
                origin="lower",
                cmap="jet",
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
        ax.set_title(f"Unit {i}", fontsize=12)
        ax.set_xlabel("X bin")
        ax.set_ylabel("Y bin")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(f"{individual_dir}/unit_{i:04d}.png", dpi=dpi, bbox_inches="tight")
        plt.close()

        if (i + 1) % 200 == 0:
            print(f"  已处理 {i + 1}/{n_units} 个单元")

    print(f"所有单独图片已保存到 {individual_dir}/")


def _plot_sample(ratemaps, output_dir, num_units, max_display=None, dpi=100):
    """随机采样模式：随机选择 N 个单元进行可视化。"""
    n_units = ratemaps.shape[0]
    max_display = min(max_display or 64, n_units)

    np.random.seed(42)
    sampled_indices = np.random.choice(n_units, size=max_display, replace=False)
    sampled_indices = np.sort(sampled_indices)

    sampled_ratemaps = ratemaps[sampled_indices]

    n_cols = min(8, max_display)
    n_rows = int(np.ceil(max_display / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(2.5 * n_cols, 2.5 * n_rows), squeeze=False
    )
    axes = axes.flatten()

    for idx, (i, ratemap) in enumerate(zip(sampled_indices, sampled_ratemaps)):
        vmin = np.nanmin(ratemap)
        vmax = np.nanmax(ratemap)

        if vmin == vmax:
            im = axes[idx].imshow(
                ratemap.T, origin="lower", cmap="gray", interpolation="nearest"
            )
        else:
            im = axes[idx].imshow(
                ratemap.T,
                origin="lower",
                cmap="jet",
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
        axes[idx].set_title(f"Unit {i}", fontsize=8)
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])

    for i in range(max_display, len(axes)):
        axes[i].axis("off")

    sampled_list = ", ".join(map(str, sampled_indices[:20]))
    if max_display > 20:
        sampled_list += f", ... (+{max_display - 20} more)"
    plt.suptitle(
        f"Sampled Spatial Ratemaps (Units: {sampled_list}, adaptive scale)", fontsize=10
    )
    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/spatial_ratemaps_sampled.png", dpi=dpi, bbox_inches="tight"
    )
    plt.close()
    print(f"采样图已保存到 {output_dir}/spatial_ratemaps_sampled.png")

    np.savez(
        f"{output_dir}/sampled_indices.npz",
        indices=sampled_indices,
        ratemaps=sampled_ratemaps,
    )
    print(f"采样单元列表已保存到 {output_dir}/sampled_indices.npz")


def plot_position_distribution(
    x: np.ndarray,
    y: np.ndarray,
    environment_bounds: tuple,
    n_bins: int,
    output_dir: str,
    dpi: int = 100,
):
    """
    绘制位置 (x, y) 的概率分布直方图/热力图。

    参数:
        x: x 坐标数组
        y: y 坐标数组
        environment_bounds: 环境边界 (min_x, max_x, min_y, max_y)
        n_bins: 分箱数量
        output_dir: 输出目录
        dpi: 图像分辨率
    """
    range_x = [environment_bounds[0], environment_bounds[1]]
    range_y = [environment_bounds[2], environment_bounds[3]]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    counts, xedges, yedges, im = axes[0].hist2d(
        x, y, bins=n_bins, range=[range_x, range_y], cmap="hot", cmin=1
    )
    axes[0].set_title("2D Position Histogram (Occupancy)", fontsize=12)
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    plt.colorbar(im, ax=axes[0], label="Visit Count")

    x_hist, x_bins = np.histogram(x, bins=n_bins, range=range_x)
    x_centers = (x_bins[:-1] + x_bins[1:]) / 2
    axes[1].bar(
        x_centers,
        x_hist,
        width=(range_x[1] - range_x[0]) / n_bins * 0.9,
        color="steelblue",
    )
    axes[1].set_title("X Position Distribution", fontsize=12)
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Visit Count")
    axes[1].set_xlim(range_x)

    y_hist, y_bins = np.histogram(y, bins=n_bins, range=range_y)
    y_centers = (y_bins[:-1] + y_bins[1:]) / 2
    axes[2].barh(
        y_centers,
        y_hist,
        height=(range_y[1] - range_y[0]) / n_bins * 0.9,
        color="coral",
    )
    axes[2].set_title("Y Position Distribution", fontsize=12)
    axes[2].set_xlabel("Visit Count")
    axes[2].set_ylabel("Y")
    axes[2].set_ylim(range_y)

    plt.suptitle(
        f"Position Distribution Analysis (n_bins={n_bins}, total_steps={len(x)})",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/position_distribution.png", dpi=dpi, bbox_inches="tight")
    plt.close()

    print(f"位置分布图已保存到 {output_dir}/position_distribution.png")

    np.savez(
        f"{output_dir}/position_distribution.npz",
        x=x,
        y=y,
        x_hist=x_hist,
        x_bins=x_bins,
        y_hist=y_hist,
        y_bins=y_bins,
        occupancy=counts,
        bounds=environment_bounds,
    )
    print(f"位置分布数据已保存到 {output_dir}/position_distribution.npz")


def compute_2d_autocorrelogram(ratemap):
    """
    计算 2D 空间自相关图 (Spatial Autocorrelogram)。
    处理了 ratemap 中可能包含的 NaN 值（未访问区域）。

    参数:
        ratemap (np.array): 形状为 (n_bins, n_bins) 的空间活动图

    返回:
        autocorr (np.array): 形状为 (2*n_bins-1, 2*n_bins-1) 的自相关图
    """
    # 将未访问的区域标记为 NaN
    rm = ratemap.copy()

    n_bins_y, n_bins_x = rm.shape
    autocorr = np.zeros((2 * n_bins_y - 1, 2 * n_bins_x - 1))
    autocorr[:] = np.nan

    # 遍历所有可能的空间滞后 (spatial lags)
    for shift_y in range(-n_bins_y + 1, n_bins_y):
        for shift_x in range(-n_bins_x + 1, n_bins_x):
            # 计算重叠区域
            y_start_1 = max(0, -shift_y)
            y_end_1 = min(n_bins_y, n_bins_y - shift_y)
            x_start_1 = max(0, -shift_x)
            x_end_1 = min(n_bins_x, n_bins_x - shift_x)

            y_start_2 = max(0, shift_y)
            y_end_2 = min(n_bins_y, n_bins_y + shift_y)
            x_start_2 = max(0, shift_x)
            x_end_2 = min(n_bins_x, n_bins_x + shift_x)

            # 提取重叠的两个子矩阵
            overlap_1 = rm[y_start_1:y_end_1, x_start_1:x_end_1]
            overlap_2 = rm[y_start_2:y_end_2, x_start_2:x_end_2]

            # 找到在两个子矩阵中都有效的索引（非 NaN）
            valid_idx = ~np.isnan(overlap_1) & ~np.isnan(overlap_2)
            n_valid = np.sum(valid_idx)

            # 至少需要 20 个有效的 bin 才计算相关性，否则噪声太大
            if n_valid >= 20:
                v1 = overlap_1[valid_idx]
                v2 = overlap_2[valid_idx]

                # 计算皮尔逊相关系数
                numerator = n_valid * np.sum(v1 * v2) - np.sum(v1) * np.sum(v2)
                denominator = np.sqrt(
                    (n_valid * np.sum(v1**2) - np.sum(v1) ** 2)
                    * (n_valid * np.sum(v2**2) - np.sum(v2) ** 2)
                )

                if denominator != 0:
                    autocorr[shift_y + n_bins_y - 1, shift_x + n_bins_x - 1] = (
                        numerator / denominator
                    )
                else:
                    autocorr[shift_y + n_bins_y - 1, shift_x + n_bins_x - 1] = 0.0

    return autocorr


def calculate_grid_score(autocorr):
    """
    计算网格得分 (Grid Score)。
    为了鲁棒性，该函数会在多个外环半径下进行搜索，并返回最高得分。

    参数:
        autocorr (np.array): 形状为 (2N-1, 2N-1) 的自相关图

    返回:
        max_grid_score (float): 最佳网格得分
    """
    center_y, center_x = autocorr.shape[0] // 2, autocorr.shape[1] // 2

    # 构建坐标网格，计算每个点到中心的距离
    y, x = np.ogrid[-center_y : center_y + 1, -center_x : center_x + 1]
    dist_from_center = np.sqrt(x**2 + y**2)

    max_grid_score = -2.0  # 初始化一个极小值
    angles_to_test = [30, 60, 90, 120, 150]

    # 动态扫描多个外环半径 (避免手动选取半径的误差)
    # 内环通常固定为避开中心峰 (如 3-5 个 bins)
    inner_radius = 4
    max_outer_radius = min(center_y, center_x) - 2

    if max_outer_radius <= inner_radius:
        return np.nan  # 空间太小，无法计算

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # 忽略全 NaN 切片警告

        for outer_radius in range(inner_radius + 4, max_outer_radius + 1, 2):
            # 生成环形掩码 (Ring Mask)
            ring_mask = (dist_from_center >= inner_radius) & (
                dist_from_center <= outer_radius
            )

            masked_autocorr = autocorr.copy()
            masked_autocorr[~ring_mask] = np.nan

            correlations = []
            for angle in angles_to_test:
                # 旋转掩码后的自相关图
                rotated_autocorr = rotate(
                    masked_autocorr, angle, reshape=False, order=1, cval=np.nan
                )

                # 提取两个图中都非 NaN 的重叠像素
                valid_idx = ~np.isnan(masked_autocorr) & ~np.isnan(rotated_autocorr)

                if np.sum(valid_idx) < 10:
                    correlations.append(np.nan)
                    continue

                v1 = masked_autocorr[valid_idx]
                v2 = rotated_autocorr[valid_idx]

                # 计算旋转前后的相关系数
                r = np.corrcoef(v1, v2)[0, 1]
                correlations.append(r)

            if not np.any(np.isnan(correlations)):
                # Score = min(60, 120) - max(30, 90, 150)
                score = min(correlations[1], correlations[3]) - max(
                    correlations[0], correlations[2], correlations[4]
                )
                if score > max_grid_score:
                    max_grid_score = score

    return max_grid_score if max_grid_score != -2.0 else np.nan


import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


def _process_single_unit(ratemap):
    """
    供多进程调用的单步执行函数：处理单个 ratemap，返回自相关图和网格得分。
    """
    autocorr = compute_2d_autocorrelogram(ratemap)
    score = calculate_grid_score(autocorr)
    return autocorr, score


def analyze_grid_cells(ratemaps, max_workers=None):
    """
    批量计算所有单元的自相关图和网格得分 (多进程加速版)。

    参数:
        ratemaps: 形状为 (N, n_bins, n_bins) 的速率图数组
        max_workers: 最大工作进程数。默认使用 CPU 核心数。
    """
    num_units = ratemaps.shape[0]

    # 如果没有指定进程数，默认使用所有可用的 CPU 核心
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()

    print(
        f"\n正在进行空间自相关分析并计算网格得分 ({num_units} 个单元，启动 {max_workers} 个进程加速)..."
    )

    autocorrs = []
    grid_scores = []

    # 使用 ProcessPoolExecutor 进行多进程计算
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # executor.map 会按顺序返回结果，保证与原始 ratemaps 的索引一致
        # 我们用 tqdm 包装它，直接获得一个漂亮的动态进度条
        results = list(
            tqdm(
                executor.map(_process_single_unit, ratemaps),
                total=num_units,
                desc="分析网格细胞",
            )
        )

    # 解包结果
    for autocorr, score in results:
        autocorrs.append(autocorr)
        grid_scores.append(score)

    return np.array(autocorrs), np.array(grid_scores)


def plot_best_grid_cells(ratemaps, autocorrs, grid_scores, output_dir, top_k=5):
    """绘制得分最高的前 K 个网格细胞的速率图和自相关图对比"""
    valid_idx = np.where(~np.isnan(grid_scores))[0]
    sorted_idx = valid_idx[np.argsort(grid_scores[valid_idx])[::-1]]
    top_indices = sorted_idx[:top_k]

    fig, axes = plt.subplots(top_k, 2, figsize=(8, 3.5 * top_k))
    if top_k == 1:
        axes = np.expand_dims(axes, 0)

    for row, unit_idx in enumerate(top_indices):
        score = grid_scores[unit_idx]

        # 绘制 Ratemap
        ax1 = axes[row, 0]
        im1 = ax1.imshow(
            ratemaps[unit_idx].T, origin="lower", cmap="jet", interpolation="nearest"
        )
        ax1.set_title(f"Unit {unit_idx} Ratemap")
        ax1.axis("off")
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        # 绘制 Autocorrelogram
        ax2 = axes[row, 1]
        # 自相关图通常在 [-1, 1] 之间，使用 coolwarm 色带最直观
        im2 = ax2.imshow(
            autocorrs[unit_idx].T, origin="lower", cmap="coolwarm", vmin=-1, vmax=1
        )
        ax2.set_title(f"Autocorr (Grid Score: {score:.3f})")
        ax2.axis("off")
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/top_grid_cells.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Top {top_k} 网格细胞对比图已保存。")


def analyze_head_direction_cells(
    angles, hidden_states, n_angle_bins=60, smooth_sigma=1.0
):
    """
    计算所有神经元的方向调谐曲线和平均向量长度 (MVL)。

    参数:
        angles: 形状为 (T,) 的方向角数组，范围 [-pi, pi]
        hidden_states: 形状为 (T, N) 的激活值
        n_angle_bins: 角度分箱数量，默认 60 (即每 6 度一个 bin)
        smooth_sigma: 高斯平滑系数

    返回:
        tuning_curves: 形状为 (N, n_angle_bins) 的调谐曲线
        mvls: 形状为 (N,) 的平均向量长度得分
        bin_centers: 每个 bin 的中心角度
    """
    bins = np.linspace(-np.pi, np.pi, n_angle_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    num_neurons = hidden_states.shape[1]
    tuning_curves = []
    mvls = []

    # 计算每个角度 bin 的停留次数 (Occupancy)
    occupancy, _ = np.histogram(angles, bins=bins)

    print(
        f"\n正在计算并平滑 {num_neurons} 个神经元的方向调谐曲线 (bins={n_angle_bins})..."
    )

    for i in range(num_neurons):
        activations = hidden_states[:, i]

        # 计算在每个角度 bin 内的激活总和
        act_sum, _ = np.histogram(angles, bins=bins, weights=activations)

        # 计算平均放电率 (Rate = Sum / Occupancy)
        with np.errstate(divide="ignore", invalid="ignore"):
            tuning_curve = np.divide(act_sum, occupancy)

        # 将未访问的角度区域(NaN)补0，避免平滑时报错
        tuning_curve[np.isnan(tuning_curve)] = 0.0

        # 对调谐曲线进行循环平滑 (mode='wrap' 保证在 -pi 和 pi 处首尾相接平滑)
        smoothed_curve = gaussian_filter1d(
            tuning_curve, sigma=smooth_sigma, mode="wrap"
        )
        tuning_curves.append(smoothed_curve)

        # ================= 计算平均向量长度 (MVL, Mean Vector Length) =================
        # 将 bin 的中心角度转换为复数形式 exp(i * theta)
        complex_angles = np.exp(1j * bin_centers)
        if np.sum(smoothed_curve) > 0:
            # 向量加权和 除以 标量总和
            mean_vector = np.sum(smoothed_curve * complex_angles) / np.sum(
                smoothed_curve
            )
            mvl = np.abs(mean_vector)  # 取复数的模长作为 MVL 得分
        else:
            mvl = 0.0

        mvls.append(mvl)

    return np.array(tuning_curves), np.array(mvls), bin_centers


def plot_best_hd_cells(tuning_curves, mvls, bin_centers, output_dir, top_k=5):
    """
    绘制 MVL 得分最高的前 K 个方向细胞的极坐标调谐图。
    """
    valid_idx = np.where(~np.isnan(mvls))[0]
    if len(valid_idx) == 0:
        return

    # 按 MVL 得分降序排序
    sorted_idx = valid_idx[np.argsort(mvls[valid_idx])[::-1]]
    top_indices = sorted_idx[:top_k]

    # 使用 polar 投影绘制极坐标图
    fig, axes = plt.subplots(
        1, top_k, figsize=(3 * top_k, 3), subplot_kw={"projection": "polar"}
    )
    if top_k == 1:
        axes = [axes]

    for ax, unit_idx in zip(axes, top_indices):
        score = mvls[unit_idx]
        curve = tuning_curves[unit_idx]

        # 极坐标图中，为了让线条闭合，需要把首个点的数据追加到末尾
        plot_angles = np.append(bin_centers, bin_centers[0])
        plot_curve = np.append(curve, curve[0])

        # 绘制并填充
        ax.plot(plot_angles, plot_curve, color="#1f77b4", linewidth=2)
        ax.fill(plot_angles, plot_curve, color="#1f77b4", alpha=0.3)

        # 美化图表
        ax.set_title(f"Unit {unit_idx}\nMVL: {score:.3f}", pad=15)
        ax.set_rticks([])  # 隐藏径向刻度，使图表更清爽
        ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/top_hd_cells.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Top {top_k} 方向细胞(HD)极坐标图已保存到 {output_dir}/top_hd_cells.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="分析 rollout data 生成空间活动图")
    parser.add_argument(
        "--data-dir", type=str, required=True, help="数据目录"
    )
    parser.add_argument("--output-dir", type=str, default="", help="输出目录")
    parser.add_argument("--n-bins", type=int, default=32, help="网格大小")
    parser.add_argument(
        "--bounds",
        type=float,
        nargs=4,
        default=None,
        metavar=("MIN_X", "MAX_X", "MIN_Y", "MAX_Y"),
        help="环境边界 (min_x, max_x, min_y, max_y)",
    )
    parser.add_argument(
        "--max-episodes", type=int, default=None, help="最大处理的 episode 数量"
    )
    parser.add_argument(
        "--plot-mode",
        type=str,
        default="grid",
        choices=["grid", "individual", "sample"],
        help="可视化模式: grid(网格拼图), individual(单独保存), sample(随机采样)",
    )
    parser.add_argument(
        "--max-display",
        type=int,
        default=None,
        help="最大显示数量 (用于 grid 和 sample 模式)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="图像分辨率",
    )

    args = parser.parse_args()

    analyze_rollout_data(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_bins=args.n_bins,
        environment_bounds=tuple(args.bounds) if args.bounds else None,
        max_episodes=args.max_episodes,
        plot_mode=args.plot_mode,
        max_display=args.max_display,
        dpi=args.dpi,
    )
