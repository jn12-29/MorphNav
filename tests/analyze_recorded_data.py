"""
Spatial Analysis Script for Recorded Data
==========================================
使用 DataRecorder 加载 recorded_data 中的数据，生成空间活动图。

参考: generate_spatiai_ratemaps.py 中的 spatial ratemap 生成方法
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
from pathlib import Path
import sys
from tqdm import tqdm

sys.path.append("/home/xh/ai4neuron/MorphNav/rl-baselines3-zoo")
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


def analyze_recorded_data(
    data_dir: str = "./recorded_data",
    output_dir: str = "./analysis_results",
    n_bins: int = 32,
    environment_bounds: tuple = None,
    max_episodes: int = None,
    plot_mode: str = "grid",
    max_display: int = None,
    dpi: int = 100,
):
    """
    分析 recorded_data 中的所有 episode，生成空间活动图。

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

    ratemaps = generate_spatial_ratemaps(
        all_x,
        all_y,
        combined_activations,
        n_bins=n_bins,
        environment_bounds=environment_bounds,
    )

    print(f"\n生成 {ratemaps.shape[0]} 个空间活动图，形状: {ratemaps.shape[1:]}")

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

    np.savez(
        f"{output_dir}/analysis_data.npz",
        x=all_x,
        y=all_y,
        ratemaps=ratemaps if combined_activations is not None else None,
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="分析 recorded_data 生成空间活动图")
    parser.add_argument(
        "--data-dir", type=str, default="./recorded_data", help="数据目录"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./analysis_results", help="输出目录"
    )
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

    analyze_recorded_data(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_bins=args.n_bins,
        environment_bounds=tuple(args.bounds) if args.bounds else None,
        max_episodes=args.max_episodes,
        plot_mode=args.plot_mode,
        max_display=args.max_display,
        dpi=args.dpi,
    )
