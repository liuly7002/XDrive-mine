import math
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
from tqdm import tqdm
import json

"""
Dense Risk Map Version (Re-designed Kernel)
===========================================
相对于原始版本的改动：
1. 不再使用原始“单一椭圆高斯 + 航向差 + 车头sigmoid”形式；
2. 改为“前后非对称双区风险核”：
   - 前向区域扩散更远；
   - 后向区域扩散更短；
   - 加入侧向增强项；
3. 航向项改为新的 sin^2(delta/2) 耦合形式；
4. 保持 dense risk map 的整体生成、可视化和 JSON 保存流程不变。

说明：
- 如果你担心 192x192 太密、文件太大，可以把 USE_DOWNSAMPLED_GRID=True，
  然后设置 DENSE_OUT_H / DENSE_OUT_W，比如 96x96 或 64x64。
"""

# ========================
# 全局开关
# ========================
DEBUG = False
index = 144
SingleFrame = False
SaveImage = True
SaveJson = True

# 是否使用降采样的 dense grid
# False -> 直接按 bev_left 原分辨率输出（例如 192x192）
# True  -> 输出指定大小（例如 96x96）
USE_DOWNSAMPLED_GRID = False
DENSE_OUT_H = 64
DENSE_OUT_W = 64

# 本地
data_root = "/home/liulei/ll/XDrive-mine/data"
root_dirs = ["/home/liulei/ll/XDrive-mine/data/kuangshan_00"]

# 服务器
# data_root = "/home/kemove/ll/PPBEV_ll/data"
# root_dirs = [
#     "/home/kemove/ll/PPBEV_ll/data/town01_original",
#     "/home/kemove/ll/PPBEV_ll/data/town03_original",
#     "/home/kemove/ll/PPBEV_ll/data/town04_original",
#     "/home/kemove/ll/PPBEV_ll/data/town06_original",
#     "/home/kemove/ll/PPBEV_ll/data/town07_original",
#     "/home/kemove/ll/PPBEV_ll/data/town10_original",
#     "/home/kemove/ll/PPBEV_ll/data/town02_val",
#     "/home/kemove/ll/PPBEV_ll/data/town05_val",
# ]

# ========================
# 新型非对称风险核函数
# ========================
def asymmetric_dual_zone_risk(points, pos, yaw, Lpix, Wpix,
                              lambda_front=1.0,
                              lambda_rear=0.45,
                              eta_lat=0.35):
    """
    新风险核：
    1) 前向区域和后向区域分别建模
    2) 前后纵向扩散不同
    3) 额外加入侧向增强项

    参数:
        points: (N, 2)
        pos: (2,)
        yaw: float
        Lpix, Wpix: 车辆在 BEV 中的长度宽度尺度
    返回:
        risk_kernel: (N,)
    """
    rel = points - pos

    c, s = np.cos(yaw), np.sin(yaw)

    # 车辆前向单位向量
    e_parallel = np.array([c, s], dtype=np.float32)
    # 车辆横向单位向量
    e_perp = np.array([-s, c], dtype=np.float32)

    # 局部坐标
    d_parallel = rel @ e_parallel
    d_perp = rel @ e_perp

    # 前向区域：沿车头方向扩散更远
    sigma_f_parallel = max(1e-3, 1.8 * Lpix)
    sigma_f_perp     = max(1e-3, 1.15 * Wpix)

    # 后向区域：纵向扩散更短，表示车尾影响衰减更快
    sigma_r_parallel = max(1e-3, 0.8 * Lpix)
    sigma_r_perp     = max(1e-3, 0.95 * Wpix)

    front_mask = (d_parallel >= 0).astype(np.float32)
    rear_mask  = 1.0 - front_mask

    front_risk = np.exp(
        -0.5 * (
            (d_parallel ** 2) / (sigma_f_parallel ** 2) +
            (d_perp ** 2) / (sigma_f_perp ** 2)
        )
    )

    rear_risk = np.exp(
        -0.5 * (
            (d_parallel ** 2) / (sigma_r_parallel ** 2) +
            (d_perp ** 2) / (sigma_r_perp ** 2)
        )
    )

    # 前后非对称组合
    base_risk = lambda_front * front_mask * front_risk + lambda_rear * rear_mask * rear_risk

    # 侧向增强：强调车辆两侧邻域的风险带
    sigma_lat = max(1e-3, 1.4 * Wpix)
    lateral_boost = 1.0 + eta_lat * np.exp(-0.5 * (d_perp ** 2) / (sigma_lat ** 2))

    return base_risk * lateral_boost


# ========================
# 稠密网格生成
# ========================
def generate_dense_bev_grid(H, W):
    """
    生成整张 BEV 左半图上的稠密网格坐标
    返回:
        grid_points: (H*W, 2)，每行是 [x, y]
    """
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    grid_points = np.stack([xs, ys], axis=-1).reshape(-1, 2).astype(np.float32)
    return grid_points


def generate_dense_bev_grid_resized(H, W, out_h=96, out_w=96):
    """
    生成降采样后的稠密网格坐标
    返回:
        grid_points: (out_h*out_w, 2)，坐标仍映射到原始 bev_left 像素平面
    """
    xs = np.linspace(0, W - 1, out_w)
    ys = np.linspace(0, H - 1, out_h)
    xs, ys = np.meshgrid(xs, ys)
    grid_points = np.stack([xs, ys], axis=-1).reshape(-1, 2).astype(np.float32)
    return grid_points


# ========================
# 贪婪匹配（原脚本保留，当前未使用）
# ========================
def _build_tracks_by_greedy_matching(history_frames, max_match_dist=50.0):
    """
    history_frames: [[bev_x, bev_y, yaw, world_length, world_width, length_pix, width_pix], ...]
    输出: tracks = [ [vehicle0], [vehicle1], ... ]
    """
    if len(history_frames) > 0 and isinstance(history_frames[0][0], (int, float)):
        frames = [history_frames]
    else:
        frames = history_frames

    Tplus1 = len(frames)
    tracks = []

    if Tplus1 == 0 or len(frames[0]) == 0:
        return tracks

    if DEBUG:
        for v in frames[index]:
            tracks.append([v])
    else:
        for v in frames:
            tracks.append([v])

    return tracks


# ========================
# 稠密风险建模
# ========================
def compute_dense_risk_with_multi_frames(points, vehicles_frames, ego_yaw, weights, scale=1.2):
    """
    融合当前帧 + 未来帧的 dense risk
    参数:
    - points: np.ndarray, (N,2)，稠密网格坐标
    - vehicles_frames: list，长度 T，每个元素是某一帧的车辆信息列表
    - ego_yaw: float，自车航向角
    - weights: list 或 np.ndarray，长度 T，时间权重
    - scale: float，风险核尺度参数
    返回:
    - risks: np.ndarray, (N,)
    """
    assert len(vehicles_frames) == len(weights), "⚠️ 帧数和权重长度必须一致"
    N = points.shape[0]
    risks_agg = np.zeros(N, dtype=float)

    for vehicles, w in zip(vehicles_frames, weights):
        risk_t = compute_dense_risk_with_single_frame(points, vehicles, ego_yaw=ego_yaw, scale=scale)
        risks_agg += w * risk_t

    # 百分位归一化
    p_max = np.percentile(risks_agg, 99.5)
    risks_agg = np.clip(risks_agg / max(p_max, 1e-12), 0, 1)
    risks_agg = np.maximum(risks_agg, 1e-6)

    return risks_agg


def compute_dense_risk_with_single_frame(points, vehicles_frame, ego_yaw, alpha=0.5, scale=1.2, eps=1e-12):
    """
    对稠密二维坐标点计算风险值，考虑：
    1) 前后非对称双区风险核
    2) ego vs 周车 航向角差耦合
    3) 侧向增强项

    参数:
    - points: (N,2)
    - vehicles_frame: 当前帧所有车辆信息
    - ego_yaw: float
    - scale: 风险场整体缩放
    """
    N = points.shape[0]
    if len(vehicles_frame) == 0:
        return np.zeros(N, dtype=float)

    P = np.zeros(N, dtype=float)

    for v in vehicles_frame:
        if len(v) < 8:
            print(f"⚠️ 跳过异常车辆信息: {v}")
            continue

        # 解包
        x_pix, y_pix, _, yaw_j, _, _, L_pix, W_pix = v
        pos = np.array([x_pix, y_pix], dtype=np.float32)

        # --- 同向 & 同车道 & 在后方 -> 跳过 ---
        ego_x, ego_y, ego_yaw_val = 96, 152, ego_yaw
        lane_thresh_x = 20
        back_offset = 10
        yaw_thresh = np.pi / 12

        if abs((yaw_j - ego_yaw_val) % (2 * np.pi) - np.pi) < yaw_thresh:
            if abs(x_pix - ego_x) < lane_thresh_x:
                if y_pix > ego_y + back_offset:
                    continue

        # 新型非对称风险核参数
        Lpix = max(1e-3, L_pix * scale)
        Wpix = max(1e-3, W_pix * scale)

        # 1) 非对称双区风险核
        g_new = asymmetric_dual_zone_risk(
            points=points,
            pos=pos,
            yaw=yaw_j,
            Lpix=Lpix,
            Wpix=Wpix,
            lambda_front=1.0,
            lambda_rear=0.45,
            eta_lat=0.35
        )

        # 2) 新的航向耦合项
        delta = ego_yaw - yaw_j
        orient_coupling = 1.0 + 0.8 * (np.sin(delta / 2.0) ** 2)

        # 3) 最终叠加
        P += g_new * orient_coupling

    if SingleFrame:
        P = P / (P.max() + eps)

    return P


# ========================
# 时间权重生成器
# ========================
def get_time_weights(num_frames, mode="manual", manual_weights=None, decay_rate=0.7):
    if mode == "manual":
        assert manual_weights is not None, "manual 模式需要提供 manual_weights"
        weights = np.array(manual_weights, dtype=float)
    elif mode == "uniform":
        weights = np.ones(num_frames, dtype=float)
    elif mode == "exp_decay":
        weights = np.array([decay_rate ** i for i in range(num_frames)], dtype=float)
    else:
        raise ValueError("mode 必须是 'manual' / 'uniform' / 'exp_decay'")

    if weights.sum() > 1e-12:
        weights = weights / weights.sum()
    return weights


# ========================
# 主处理函数
# ========================
def process_one(
    front_img_path,
    vehicles_info,
    ego_pos=(96, 152),
    ego_yaw=0.0,
    scale=2.0,
    r_max=100,
    future_vehicles_info=None,
    time_weights=None
):
    if isinstance(front_img_path, list):
        front_img_path = front_img_path[0]

    rel_path = front_img_path.lstrip("/")
    bev_path = os.path.join(data_root, rel_path.replace("rgb", "bev"))

    bev_img = np.array(Image.open(bev_path))
    H, W, C = bev_img.shape
    bev_left = bev_img[:, :W // 2, :]
    H_left, W_left, _ = bev_left.shape

    # =========================
    # 1) 生成 dense grid
    # =========================
    if USE_DOWNSAMPLED_GRID:
        grid_points = generate_dense_bev_grid_resized(H_left, W_left, DENSE_OUT_H, DENSE_OUT_W)
        out_h, out_w = DENSE_OUT_H, DENSE_OUT_W
    else:
        grid_points = generate_dense_bev_grid(H_left, W_left)
        out_h, out_w = H_left, W_left

    # =========================
    # 2) 计算 dense risk
    # =========================
    if future_vehicles_info is not None and time_weights is not None:
        vehicles_frames = [vehicles_info] + list(future_vehicles_info)
        risks = compute_dense_risk_with_multi_frames(
            grid_points, vehicles_frames, ego_yaw=ego_yaw,
            weights=time_weights, scale=scale
        )
    else:
        risks = compute_dense_risk_with_single_frame(
            grid_points, vehicles_info, ego_yaw=ego_yaw, scale=scale
        )

    risk_map = risks.reshape(out_h, out_w)

    # =========================
    # 3) 后处理
    # =========================
    if (out_h == H_left) and (out_w == W_left):
        risks_clamped = risk_map.copy()

        # 背景区域强制高风险
        background_mask = np.all(bev_left == 0, axis=-1)
        risks_clamped[background_mask] = 1.0

        risk_map = risks_clamped
    else:
        risks_clamped = risks.copy()
        for i, (px, py) in enumerate(grid_points.astype(int)):
            if 0 <= px < W_left and 0 <= py < H_left:
                pixel = bev_left[py, px]
                if np.all(pixel == 0):
                    risks_clamped[i] = 1.0
        risk_map = risks_clamped.reshape(out_h, out_w)

    # =========================
    # 4) 保存可视化图
    # =========================
    if SaveImage:
        cmap = plt.get_cmap('RdYlGn_r')

        # ---- 4.1 在 bev_left 上保存 dense heatmap overlay ----
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(bev_left)
        ax.imshow(
            risk_map,
            cmap=cmap,
            alpha=0.6,
            vmin=0,
            vmax=1,
            origin='upper',
            extent=[0, W_left, H_left, 0]
        )
        ax.axis('off')

        save_dir = os.path.join(os.path.dirname(os.path.dirname(bev_path)), "bev_dense_risk")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, os.path.basename(bev_path))
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close()

        # ---- 4.2 在 bev_rgb 上保存 dense heatmap overlay ----
        bev_rgb_path = bev_path.replace(os.sep + "bev" + os.sep, os.sep + "bev_rgb" + os.sep)

        if os.path.exists(bev_rgb_path):
            bev_rgb_img = np.array(Image.open(bev_rgb_path))
            H_rgb, W_rgb, _ = bev_rgb_img.shape

            # 几何参数（沿用你原脚本）
            PIXELS_PER_METER = 2.0
            BEV_WIDTH = 192
            PIXELS_EV_TO_BOTTOM = 40

            ego_x_bev, ego_y_bev = BEV_WIDTH / 2.0, BEV_WIDTH - PIXELS_EV_TO_BOTTOM

            h_cam = 70.0
            fov_deg = 50.0
            R_rgb = h_cam * math.tan(math.radians(fov_deg / 2.0))
            meters_per_pix_rgb = (2.0 * R_rgb) / W_rgb
            ego_x_rgb, ego_y_rgb = W_rgb / 2.0, H_rgb / 2.0

            if USE_DOWNSAMPLED_GRID:
                xs = np.linspace(0, W_left - 1, out_w)
                ys = np.linspace(0, H_left - 1, out_h)
            else:
                xs = np.arange(W_left)
                ys = np.arange(H_left)

            xs_mesh, ys_mesh = np.meshgrid(xs, ys)

            offset_bev_x = xs_mesh - ego_x_bev
            offset_bev_y = ys_mesh - ego_y_bev

            offset_m_x = offset_bev_x / PIXELS_PER_METER
            offset_m_y = offset_bev_y / PIXELS_PER_METER

            offset_rgb_x = offset_m_x / meters_per_pix_rgb
            offset_rgb_y = offset_m_y / meters_per_pix_rgb

            x_rgb = ego_x_rgb + offset_rgb_x
            y_rgb = ego_y_rgb + offset_rgb_y

            fig = plt.figure(figsize=(W_rgb / 100.0, H_rgb / 100.0), dpi=100)
            ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
            ax.imshow(bev_rgb_img)

            ax.imshow(
                risk_map,
                cmap=cmap,
                alpha=0.45,
                vmin=0,
                vmax=1,
                origin='upper'
            )

            ax.axis('off')

            bev_rgb_save_dir = os.path.join(
                os.path.dirname(os.path.dirname(bev_rgb_path)),
                "bev_rgb_dense_risk"
            )
            os.makedirs(bev_rgb_save_dir, exist_ok=True)
            bev_rgb_save_path = os.path.join(
                bev_rgb_save_dir,
                os.path.basename(bev_rgb_path)
            )
            plt.savefig(bev_rgb_save_path, dpi=100, bbox_inches='tight', pad_inches=0)
            plt.close()
        else:
            print(f"[Warn] 对应的 bev_rgb 不存在: {bev_rgb_path}")

    # =========================
    # 5) 保存 JSON
    # =========================
    if SaveJson:
        json_save_dir = os.path.join(os.path.dirname(os.path.dirname(bev_path)), "dense_risk_labels")
        os.makedirs(json_save_dir, exist_ok=True)
        json_save_path = os.path.join(json_save_dir, os.path.basename(bev_path).replace(".png", ".json"))

        with open(json_save_path, 'w') as f:
            json.dump(risk_map.tolist(), f)


# ========================
# 主入口
# ========================
if __name__ == "__main__":
    if DEBUG:
        file_name = "/home/liulei/ll/PPBEV_ll/data/town01_val/routes_town01_val_12_04_11_10_40/packed_data.npy"
        data = np.load(file_name, allow_pickle=True).item()

        front_img_path = data["front_img"][index]
        vehicles_info = data["vehicles_info"][index]
        future_vehicles_info = data["future_vehicles_info"][index]
        ego_yaw = np.array(math.pi / 2)

        weight_mode = "exp_decay"
        if weight_mode == "manual":
            time_weights = get_time_weights(
                num_frames=1 + len(future_vehicles_info),
                mode="manual",
                manual_weights=[0.5, 0.25, 0.15, 0.1, 0.0]
            )
        elif weight_mode == "uniform":
            time_weights = get_time_weights(
                num_frames=1 + len(future_vehicles_info),
                mode="uniform"
            )
        elif weight_mode == "exp_decay":
            time_weights = get_time_weights(
                num_frames=1 + len(future_vehicles_info),
                mode="exp_decay",
                decay_rate=0.7
            )

        process_one(
            front_img_path,
            vehicles_info,
            ego_yaw=ego_yaw,
            scale=2.0,
            r_max=100,
            future_vehicles_info=future_vehicles_info,
            time_weights=time_weights
        )

    else:
        for root_dir in root_dirs:
            print(f"========== 正在处理 root_dir = {root_dir} ==========")
            data_paths = glob.glob(os.path.join(root_dir, "*", "packed_data.npy"))

            for data_path in data_paths:
                data = np.load(data_path, allow_pickle=True).item()

                front_imgs = data["front_img"]
                vehicles_infos = data["vehicles_info"]
                ego_yaw = np.array(math.pi / 2)

                for idx, (front_img_path, vehicles_info) in enumerate(
                    tqdm(
                        zip(front_imgs, vehicles_infos),
                        total=len(front_imgs),
                        desc=f" 🐱 {os.path.basename(root_dir)}",
                        leave=False
                    )
                ):
                    future_vehicles_info = data["future_vehicles_info"][idx]

                    weight_mode = "exp_decay"
                    if weight_mode == "manual":
                        time_weights = get_time_weights(
                            num_frames=1 + len(future_vehicles_info),
                            mode="manual",
                            manual_weights=[0.5, 0.25, 0.15, 0.1, 0.0]
                        )
                    elif weight_mode == "uniform":
                        time_weights = get_time_weights(
                            num_frames=1 + len(future_vehicles_info),
                            mode="uniform"
                        )
                    elif weight_mode == "exp_decay":
                        time_weights = get_time_weights(
                            num_frames=1 + len(future_vehicles_info),
                            mode="exp_decay",
                            decay_rate=0.7
                        )

                    if SingleFrame:
                        process_one(
                            front_img_path,
                            vehicles_info,
                            ego_yaw=ego_yaw,
                            scale=2.0,
                            r_max=100,
                            future_vehicles_info=None,
                            time_weights=None
                        )
                    else:
                        process_one(
                            front_img_path,
                            vehicles_info,
                            ego_yaw=ego_yaw,
                            scale=2.0,
                            r_max=100,
                            future_vehicles_info=future_vehicles_info,
                            time_weights=time_weights
                        )