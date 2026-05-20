import numpy as np
import matplotlib

matplotlib.use("Agg")  # 强制使用无交互后端，加速渲染并保证多进程线程安全
import matplotlib.pyplot as plt
import argparse
import os
import imageio
from concurrent.futures import ProcessPoolExecutor
import multiprocessing


def load_and_align_data(
    npz_path: str, env_idx: int, max_frames: int = 0, reset_thresh: float = 0.0
):
    data = np.load(npz_path)
    pred_pos = data["pred_pos"]
    true_pos = data["true_pos"]

    if len(true_pos) == 0:
        raise ValueError(
            "数据中未找到 true_pos，请检查保存步骤中是否正确提取了真实坐标。"
        )

    if pred_pos.ndim == 3:
        if env_idx >= pred_pos.shape[1]:
            raise ValueError(
                f"指定的 env_idx ({env_idx}) 超出范围，总环境数为 {pred_pos.shape[1]}"
            )
        p_pos = pred_pos[:, env_idx, :]
        t_pos = true_pos[:, env_idx, :]
    elif pred_pos.ndim == 2:
        p_pos = pred_pos
        t_pos = true_pos
    else:
        raise ValueError(f"预期数据维度为 2 或 3，但得到了: pred_pos {pred_pos.shape}")

    if max_frames > 0 and len(t_pos) > max_frames:
        print(
            f"[数据截断] 当前轨迹长度 ({len(t_pos)}) 超过设定的最大帧数 ({max_frames})，已自动截断。"
        )
        t_pos = t_pos[:max_frames]
        p_pos = p_pos[:max_frames]

    # --- 核心新增：基于真实位置的环境重置(突变)检测与断线处理 ---
    if len(t_pos) > 1:
        step_dists = np.linalg.norm(t_pos[1:] - t_pos[:-1], axis=1)

        if reset_thresh > 0:
            threshold = reset_thresh
        else:
            # 自动阈值计算：利用四分位距(IQR)识别极端离群值，结合硬下限规避极小步长误判
            q75, q25 = np.percentile(step_dists, [75, 25])
            iqr = q75 - q25
            threshold = max(q75 + 10 * iqr, 2.0)

        split_indices = np.where(step_dists > threshold)[0] + 1

        if len(split_indices) > 0:
            print(
                f"[数据处理] 检测到 {len(split_indices)} 次环境重置(位置突变，阈值={threshold:.2f})，已执行断点隔离。"
            )
            # 在突变位置插入 np.nan，Matplotlib 遇到 nan 会自动断开连线
            t_pos = np.insert(t_pos, split_indices, np.nan, axis=0)
            p_pos = np.insert(p_pos, split_indices, np.nan, axis=0)

    return t_pos, p_pos


def visualize_positions(
    npz_path: str,
    env_idx: int = 0,
    save_path: str = None,
    max_frames: int = 0,
    reset_thresh: float = 0.0,
):
    t_pos, p_pos = load_and_align_data(npz_path, env_idx, max_frames, reset_thresh)
    # 忽略 NaN 值计算 MSE
    mse_loss = np.nanmean((t_pos - p_pos) ** 2)

    plt.figure(figsize=(10, 10))
    plt.plot(
        t_pos[:, 0],
        t_pos[:, 1],
        label="True Position",
        color="#1f77b4",
        alpha=0.7,
        linewidth=2,
        marker="o",
        markersize=3,
    )
    plt.plot(
        p_pos[:, 0],
        p_pos[:, 1],
        label="Predicted Position (Aux Head)",
        color="#ff7f0e",
        alpha=0.7,
        linewidth=2,
        linestyle="--",
        marker="x",
        markersize=3,
    )

    # 提取有效起点和终点
    valid_mask = ~np.isnan(t_pos[:, 0])
    first_idx = np.where(valid_mask)[0][0]
    last_idx = np.where(valid_mask)[0][-1]

    plt.scatter(
        *t_pos[first_idx], color="green", s=150, zorder=5, marker="*", label="Start"
    )
    plt.scatter(*t_pos[last_idx], color="red", s=150, zorder=5, marker="s", label="End")
    plt.scatter(
        *p_pos[first_idx], color="green", s=150, zorder=5, marker="*", alpha=0.5
    )
    plt.scatter(*p_pos[last_idx], color="red", s=150, zorder=5, marker="s", alpha=0.5)

    sample_rate = max(1, len(t_pos) // 50)
    for i in range(0, len(t_pos), sample_rate):
        if not np.isnan(t_pos[i, 0]) and not np.isnan(p_pos[i, 0]):
            plt.plot(
                [t_pos[i, 0], p_pos[i, 0]],
                [t_pos[i, 1], p_pos[i, 1]],
                color="gray",
                alpha=0.3,
                linestyle=":",
                linewidth=1,
            )

    plt.text(
        0.05,
        0.95,
        f"MSE Loss: {mse_loss:.6f}\nFrames: {len(t_pos)}",
        transform=plt.gca().transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"),
    )

    plt.title(
        f"Trajectory & Localization Prediction (Env {env_idx})", fontsize=16, pad=15
    )
    plt.xlabel("X Coordinate", fontsize=12)
    plt.ylabel("Y Coordinate", fontsize=12)
    plt.legend(loc="upper right", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.axis("equal")

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"静态可视化图像已保存至: {save_path}")
        plt.close()
    else:
        plt.show()


# --- 独立出渲染单帧的函数，以便多进程序列化 (Pickling) ---
def _render_single_frame(args):
    frame, t_pos, p_pos, env_idx, mse_loss, total_frames, x_lim, y_lim, sample_rate = (
        args
    )

    fig, ax = plt.subplots(
        figsize=(8, 8), dpi=100
    )  # 调低 DPI 以进一步加速并减小内存占用
    ax.set_title(f"Dynamic Trajectory (Env {env_idx})", fontsize=16, pad=15)
    ax.set_xlabel("X Coordinate", fontsize=12)
    ax.set_ylabel("Y Coordinate", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_aspect("equal", adjustable="box")

    # 绘制轨迹到当前帧
    if frame > 0:
        ax.plot(
            t_pos[:frame, 0],
            t_pos[:frame, 1],
            label="True Position",
            color="#1f77b4",
            alpha=0.7,
            linewidth=2,
        )
        ax.plot(
            p_pos[:frame, 0],
            p_pos[:frame, 1],
            label="Predicted Position",
            color="#ff7f0e",
            alpha=0.7,
            linewidth=2,
            linestyle="--",
        )

        # 绘制误差虚线
        for i in range(0, frame, sample_rate):
            if not np.isnan(t_pos[i, 0]) and not np.isnan(p_pos[i, 0]):
                ax.plot(
                    [t_pos[i, 0], p_pos[i, 0]],
                    [t_pos[i, 1], p_pos[i, 1]],
                    color="gray",
                    alpha=0.3,
                    linestyle=":",
                    linewidth=1,
                )

    valid_mask = ~np.isnan(t_pos[:, 0])
    if valid_mask.any():
        first_idx = np.where(valid_mask)[0][0]
        ax.scatter(
            *t_pos[first_idx], color="green", s=150, zorder=5, marker="*", label="Start"
        )

    ax.legend(loc="upper right", fontsize=11)

    ax.text(
        0.05,
        0.95,
        f"MSE Loss: {mse_loss:.6f}\nFrames: {frame}/{total_frames}",
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"),
    )

    # 将画布直接提取为 Numpy RGB 数组
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[..., :3]  # 仅保留 RGB 丢弃 Alpha
    plt.close(fig)
    return image


def animate_positions_parallel(
    npz_path: str,
    env_idx: int = 0,
    save_path: str = None,
    max_frames: int = 0,
    reset_thresh: float = 0.0,
):
    t_pos, p_pos = load_and_align_data(npz_path, env_idx, max_frames, reset_thresh)
    mse_loss = np.nanmean((t_pos - p_pos) ** 2)

    total_frames = len(t_pos)
    target_duration = 5.0
    auto_fps = 100

    print(
        f"[动画参数] 渲染帧数: {total_frames}, 自动分配帧率: {auto_fps} FPS, 预计生成时长: {total_frames/auto_fps:.1f} 秒"
    )

    # 预计算统一的坐标系边界 (使用 nanmax/nanmin 规避 nan 影响)
    all_x = np.concatenate([t_pos[:, 0], p_pos[:, 0]])
    all_y = np.concatenate([t_pos[:, 1], p_pos[:, 1]])
    margin_x = (np.nanmax(all_x) - np.nanmin(all_x)) * 0.1
    margin_y = (np.nanmax(all_y) - np.nanmin(all_y)) * 0.1
    x_lim = (np.nanmin(all_x) - margin_x, np.nanmax(all_x) + margin_x)
    y_lim = (np.nanmin(all_y) - margin_y, np.nanmax(all_y) + margin_y)

    sample_rate = max(1, total_frames // 50)

    # 准备多进程任务参数
    tasks = [
        (
            frame,
            t_pos,
            p_pos,
            env_idx,
            mse_loss,
            total_frames,
            x_lim,
            y_lim,
            sample_rate,
        )
        for frame in range(1, total_frames + 1)
    ]

    print(
        f"[并行加速] 正在启动进程池 (CPU 核心数: {multiprocessing.cpu_count()}) ... 渲染中"
    )

    frames_rgb = []
    # 使用进程池并行渲染图像
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        # executor.map 保证返回顺序与任务顺序一致
        for img in executor.map(_render_single_frame, tasks):
            frames_rgb.append(img)
            if len(frames_rgb) % 50 == 0:
                print(f"已渲染 {len(frames_rgb)} / {total_frames} 帧...")

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        # 使用 imageio 高效写出 MP4，移除 loop 参数，增加 macro_block_size=None 防止非偶数像素报错
        imageio.mimsave(save_path, frames_rgb, fps=auto_fps, macro_block_size=None)
        print(f"动态可视化已保存至: {save_path}")
    else:
        print("未指定保存路径，跳过 MP4 导出。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize and animate predicted vs true positions from npz file (Parallelized & Reset Aware)."
    )
    parser.add_argument(
        "-data-dir",
        "--data-dir",
        type=str,
        required=True,
        help="Path to the rollout data directory containing positions.npz",
    )
    parser.add_argument(
        "--env-idx", type=int, default=0, help="Environment index to plot (default: 0)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=3000,
        help="如果总帧数超过此值则截断 (默认: 1000，设置为 0 则不截断)",
    )
    parser.add_argument(
        "--reset-thresh",
        type=float,
        default=0.4,
        help="强制指定RL环境重置的突变距离阈值 (默认0：基于四分位距自动判断)",
    )

    args = parser.parse_args()
    npz_file = os.path.join(args.data_dir, "positions.npz")

    if not os.path.exists(npz_file):
        print(f"错误: 找不到文件 {npz_file}")
    else:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(args.data_dir)), "analysis")
        os.makedirs(output_dir, exist_ok=True)

        static_save_path = os.path.join(output_dir, "positions_plot.png")
        visualize_positions(
            npz_file, args.env_idx, static_save_path, args.max_frames, args.reset_thresh
        )

        anim_save_path = os.path.join(output_dir, "positions_animation.mp4")
        animate_positions_parallel(
            npz_file, args.env_idx, anim_save_path, args.max_frames, args.reset_thresh
        )
