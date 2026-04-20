import math
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
from tqdm import tqdm
import json

"""
⚠️⚠️⚠️
修改1处：
    root_dir = "/home/liulei/ll/PPBEV_ll/data/town02_original"
    
如果进行极点消融实验,即修改极点的结构(列密度)，需要修改的位置包括(以25行36列为例)：
    1.generate_multi_fov_poles_angle_focus函数的"base_pts=36"
    2.save_dir = os.path.join(os.path.dirname(os.path.dirname(bev_path)), "bev_polarpoint_25x36")中的"bev_polarpoint_25x36"
    3.rows, cols = 25, 36  # 根据生成极点时的设定  中的"36"
    4.json_save_dir = os.path.join(os.path.dirname(os.path.dirname(bev_path)), "polarpoint_labels_25x36")中的"polarpoint_labels_25x36"
⚠️⚠️⚠️
"""
# For Debug
DEBUG = False           # DEBUG模式是示例(仅处理一张图像)
index = 144             # DEBUG模式下需要处理的帧号
SingleFrame = False     # True: 只根据当前帧来生成极点风险概率 False:根据当前帧和未来4帧融合来生成极点风险概率
SaveImage = True        # 是否保存带有极点表示的BEV图
SaveJson  = True        # 是否保存极点标签为.json文件

# 本地
data_root = "/home/liulei/ll/PPBEV_ll/data"
root_dirs  = ["/home/liulei/ll/PPBEV_ll/data/town05_val_00"]   # ⚠️ 修改: 需要处理的文件夹(每次修改这里)
# 服务器
# data_root = "/home/kemove/ll/PPBEV_ll/data"
# root_dirs = [                                                   # ⚠️ 修改: 需要处理的文件夹(每次修改这里)
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
# 高斯核函数
# ========================
def gaussian_density(P, mu, Sigma_inv):
    diff = P - mu
    dist = np.einsum('...i,ij,...j->...', diff, Sigma_inv, diff)
    return np.exp(-0.5 * dist)

# ========================
# 极点静态结构生成
# ========================
def generate_multi_fov_poles_angle_focus(center,
                                         fovs=[100,70,50],
                                         layers_per_fov=[18,5,2],  # layers_per_fov=[18,5,2]
                                         r_start=10,               # 全局最近层半径(像素)
                                         r_total=140,              # 全局最远层半径(像素)
                                         base_pts=36,              # 每层的极点数量 baseline(36)
                                         offset=10,                # 极点的第一层偏离自车中心位置的大小(y方向,像素)
                                         angle_focus_ratio=1.0,    # 聚焦总角度占最小FOV比例
                                         angle_focus_factor=2.0,   # 中心密集程度
                                         focus_mode="absolute",    # "relative" 或 "absolute"
                                         layer_spacing_factor=3.0  # 层间距增大因子 (>1)
                                         ):
    """
    多视角极点生成
    - 每层点从中心向两侧逐渐稀疏
    - 聚焦区域绝对角度一致，以最小 FOV 为参考
    - 所有层之间半径全局逐渐增大（近层密集，远层稀疏）
    """
    pole_list = []

    # 总层数
    total_layers = sum(layers_per_fov)

    # 生成全局递增的层间半径
    r_ratios = np.linspace(1, layer_spacing_factor, total_layers)  # 生成一个长度为 total_layers 的等差数列，从 1 逐渐增加到 layer_spacing_factor
    r_ratios = r_ratios / r_ratios.sum()  # 归一化
    dr_layers = (r_total - r_start) * r_ratios
    radii_global = r_start + np.cumsum(dr_layers)

    # 聚焦角度
    if focus_mode == "absolute":
        min_fov = min(fovs)
        focus_angle_abs = (min_fov * angle_focus_ratio) / 2
    else:
        focus_angle_abs = None

    layer_idx = 0  # 全局层索引
    for fov, num_layers in zip(fovs, layers_per_fov):
        points_per_layer = base_pts

        if focus_mode == "relative":
            u = np.linspace(-1, 1, points_per_layer)
            u = np.sign(u) * (np.abs(u) ** angle_focus_factor)
            angles = np.deg2rad(u * (fov / 2.0))
        elif focus_mode == "absolute":
            focus_angle = min(focus_angle_abs, fov/2)                  # 定义"聚焦角度"范围
            u = np.linspace(-1, 1, points_per_layer)   # 生成[-1,1]的均匀采样序列,长度为points_per_layer
            u = np.sign(u) * (np.abs(u) ** angle_focus_factor)         # 对序列u进行指数变换,使得靠近0(前方中心)的点更密集,边缘更稀疏
            angles = np.deg2rad(u * focus_angle)                       # 将归一化后的u映射到实际角度范围[-focus_angle,focus_angle],此时只覆盖聚焦角度区域

            # 如果 FOV 比聚焦角度更大，需要把聚焦区域外的角度"扩展"到整个 FOV
            # 超出聚焦角度指数扩展到 FOV 边界
            max_side_angle = fov / 2
            side_factor = max_side_angle / focus_angle
            angles = np.sign(angles) * (np.abs(angles) * side_factor)
        else:
            raise ValueError("focus_mode 必须是 'relative' 或 'absolute'")

        # 对应半径
        radii = radii_global[layer_idx : layer_idx + num_layers]

        for r in radii:
            for a in angles:
                dx = r * np.cos(a + np.pi/2)
                dy = r * np.sin(a + np.pi/2)
                pole_list.append([center[0] + dx, center[1] - dy - offset])

        layer_idx += num_layers

    return np.array(pole_list)

# ========================
# 贪婪匹配
# ========================
def _build_tracks_by_greedy_matching(history_frames, max_match_dist=50.0):
    """
    history_frames: [[bev_x, bev_y, yaw, world_length, world_width, length_pix, width_pix], ...]
    输出: tracks = [ [vehicle0], [vehicle1], ... ]
    """
    # 如果传进来的是单帧，就保证它是 list[list[...]]
    if len(history_frames) > 0 and isinstance(history_frames[0][0], (int, float)):
        # 说明直接传的是一帧车辆列表
        frames = [history_frames]
    else:
        # 兼容原来形式 [frame0]，frame0是车辆列表
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
# 极点建模
# ========================
# 多帧风险聚合(叠加多帧"单帧风险")
def compute_poles_risk_with_multi_frames(poles, vehicles_frames, ego_yaw, weights, scale=1.2):
    """
    融合当前帧 + 未来帧的极点风险
    参数:
    - poles: np.ndarray, (N,2)，极点坐标
    - vehicles_frames: list，长度 T，每个元素是某一帧的车辆信息列表
    - ego_yaw: float，自车航向角
    - weights: list 或 np.ndarray，长度 T，时间权重（手动设置）
    - scale: float，高斯缩放参数
    返回:
    - risks: np.ndarray, (N,)，融合后的风险值
    """
    assert len(vehicles_frames) == len(weights), "⚠️ 帧数和权重长度必须一致"
    N = poles.shape[0]
    risks_agg = np.zeros(N, dtype=float)

    for vehicles, w in zip(vehicles_frames, weights):
        risk_t = compute_poles_risk_with_single_frame(poles, vehicles, ego_yaw=ego_yaw, scale=scale)
        risks_agg += w * risk_t

    # # --- 全局归一化（跨场景一致性） ---
    # global_max = max(risks_agg.max(), 1e-12)  # 防止除零
    # risks_agg = risks_agg / global_max
    # --- 改进全局归一化（百分位 + 小常数平滑，适合作为训练标签） ---
    p_max = np.percentile(risks_agg, 99.5)  # 找到 risks_agg 数组中 99.5% 的极点风险值都小于或等于 p_max, 然后用 p_max 来归一化整个数组，避免极少数异常大值压缩其他标签
    risks_agg = np.clip(risks_agg / max(p_max, 1e-12), 0, 1)  # 保证 [0,1]
    risks_agg = np.maximum(risks_agg, 1e-6)  # 小常数平滑，避免训练零梯度

    return risks_agg

# 单帧风险
def compute_poles_risk_with_single_frame(poles, vehicles_frame, ego_yaw, alpha=0.5, scale=1.2, eps=1e-12):
    """
    计算每个 pole 的风险值，考虑周围车辆的椭圆高斯分布 + 航向角差 + 车头方向加权。

    参数:
    - poles: np.ndarray, shape=(N,2)，pole 在 BEV 图像中的像素坐标
    - vehicles_frame: list，每个元素包含周车的 BEV 像素位置、航向角、长宽
    - ego_yaw: float，自车在 BEV 坐标系下的航向角
    - alpha, scale: 控制高斯核形状和缩放
    - eps: 防止除零

    返回:
    - P: np.ndarray, shape=(N,)，每个 pole 的归一化风险值
    """
    N = poles.shape[0]
    if len(vehicles_frame) == 0:
        return np.zeros(N, dtype=float)

    P = np.zeros(N, dtype=float)

    for v in vehicles_frame:
        if len(v) < 8:
            print(f"⚠️ 跳过异常车辆信息: {v}")
            continue

        # 解包
        x_pix, y_pix, _, yaw_j, _, _, L_pix, W_pix = v
        pos = np.array([x_pix, y_pix])

        # --- 🔥 同向&同车道&在后方 -> 跳过 🔥 ---
        ego_x, ego_y, ego_yaw_val = 96, 152, ego_yaw  # 自车位置和朝向
        lane_thresh_x = 20          # 横向阈值，像素单位（车道宽度）
        back_offset = 10            # 后方判定的y差阈值
        yaw_thresh = np.pi / 12     # 航向角相似阈值 (15°)
        if abs((yaw_j - ego_yaw_val) % (2*np.pi) - np.pi) < yaw_thresh:
            if abs(x_pix - ego_x) < lane_thresh_x:  # 横向在阈值内（同车道）
                if y_pix > ego_y + back_offset:     # 在自车后方
                    continue                        # 直接跳过这个车辆的风险贡献

        # 高斯核
        # Lpix, Wpix = max(1e-3, L_pix * scale), max(1e-3, W_pix * scale)
        Lpix, Wpix = max(1e-3, L_pix), max(1e-3, W_pix)
        Lambda = np.diag([Lpix**2, Wpix**2])
        R = np.array([[np.cos(yaw_j), -np.sin(yaw_j)],
                      [np.sin(yaw_j),  np.cos(yaw_j)]])
        Sigma_inv = np.linalg.inv(R @ Lambda @ R.T)

        g = gaussian_density(poles, pos, Sigma_inv)

        # 航向角差 (ego vs vehicle) 区分同向行使和对向行使: 对向行使的时候是2倍的风险,同向行使是基础风险
        delta = ego_yaw - yaw_j
        orient = 1.0 + (1.0 - np.cos(delta)) / 2.0

        # --- 🔥 车头方向加权（带最小基线 beta） ---
        beta = 0.2
        rel = poles - pos  # 相对向量
        forward = np.array([np.cos(yaw_j), np.sin(yaw_j)])  # 车头方向
        proj = rel @ forward  # 投影到车头方向
        sigmoid = 1.0 / (1.0 + np.exp(-proj / (Lpix * 0.5)))
        head_weight = beta + (1 - beta) * sigmoid
        # 车尾 -> beta
        # 车中 -> ~0.5
        # 车头 -> 1.0

        # 累加风险
        P += g * orient * head_weight

    if SingleFrame:
        # 归一化风险到 [0, 1]
        P = P / (P.max() + eps)
    return P

# ========================
# 时间权重生成器
# ========================
def get_time_weights(num_frames, mode="manual", manual_weights=None, decay_rate=0.7):
    """
    生成时间权重
    参数:
    - num_frames: int，总帧数 (当前帧 + 未来帧)
    - mode: str, "manual" / "uniform" / "exp_decay"
    - manual_weights: list，手动模式下的权重
    - decay_rate: float，指数衰减的因子 (0<decay_rate<1)
    返回:
    - weights: np.ndarray, shape=(num_frames,)
    """
    if mode == "manual":       # 手动模式: 完全自己定义每一帧的权重
        assert manual_weights is not None, " manual 模式需要提供 manual_weights"
        weights = np.array(manual_weights, dtype=float)
    elif mode == "uniform":    # 均匀模式: 所有帧的权重完全相同
        weights = np.ones(num_frames, dtype=float)
    elif mode == "exp_decay":  # 指数衰减模式: 当前帧权重最大，随着时间推移，未来帧权重按指数规律递减
        weights = np.array([decay_rate**i for i in range(num_frames)], dtype=float)
    else:
        raise ValueError(" mode 必须是 'manual' / 'uniform' / 'exp_decay'")

    # 归一化到和为1
    if weights.sum() > 1e-12:
        weights = weights / weights.sum()
    return weights

# ========================
# 主函数
# ========================
def process_one(front_img_path, vehicles_info, ego_pos=(96, 152), ego_yaw=0.0, scale=2.0, r_max=100, future_vehicles_info=None, time_weights=None):

    if isinstance(front_img_path, list):
        front_img_path = front_img_path[0]

    rel_path = front_img_path.lstrip("/")
    bev_path = os.path.join(data_root, rel_path.replace("rgb", "bev"))

    bev_img = np.array(Image.open(bev_path))
    H, W, C = bev_img.shape
    bev_left = bev_img[:, :W//2, :]

    poles = generate_multi_fov_poles_angle_focus(center=np.array(ego_pos))  # 极点静态结构

    if future_vehicles_info is not None and time_weights is not None:
        vehicles_frames = [vehicles_info] + list(future_vehicles_info)
        risks = compute_poles_risk_with_multi_frames(poles, vehicles_frames, ego_yaw=ego_yaw,
                                                     weights=time_weights, scale=scale)
    else:
        # 只用当前帧
        risks = compute_poles_risk_with_single_frame(poles, vehicles_info, ego_yaw=ego_yaw, scale=scale)

    # 在后处理前检查归一化结果
    # 方便确认多帧融合后的风险是否已经在 [0,1]
    # print(f"[Debug] risks min={risks.min():.4f}, max={risks.max():.4f}")

    # === 后处理方案1(硬标签)：背景 & 实线 & 虚线 覆盖 ===
    risks_clamped = risks.copy()
    H_left, W_left, _ = bev_left.shape
    for i, (px, py) in enumerate(poles.astype(int)):
        if 0 <= px < W_left and 0 <= py < H_left:
            pixel = bev_left[py, px]
            if np.all(pixel == 0):                # 背景(非道路)
                risks_clamped[i] = 1.0            # 强制最大风险
            # if np.all(pixel == [255, 0, 255]):    # 实线
            #     risks_clamped[i] = 1.0            # 强制最大风险
            # if np.all(pixel == [255, 140, 255]):  # 虚线
            #     risks_clamped[i] = 0.9            # 强制风险为0.9
    risks = risks_clamped

    # === 后处理方案2(软标签)：背景 & 实线车道线 & 虚线车道线 覆盖（背景渐变，仅作用于非道路极点） ===
    # risks_clamped = risks.copy()
    # H_left, W_left, _ = bev_left.shape
    # # 背景到道路距离图
    # road_color = np.array([46, 52, 54])
    # road_mask = np.all(bev_left == road_color, axis=-1)
    # from scipy.ndimage import distance_transform_edt
    # dist_to_road = distance_transform_edt(~road_mask)  # 背景到道路的距离（像素）
    # # 背景渐变风险参数
    # max_dist = 10.0  # 超过这个距离视为最大风险
    # beta = 1.0       # 背景最大风险
    # alpha = 0.7      # 道路附近基线风险
    # # 计算背景风险
    # background_risk = alpha + (beta - alpha) * np.clip(dist_to_road / max_dist, 0, 1)
    # # 应用到极点
    # for i, (px, py) in enumerate(poles.astype(int)):
    #     if 0 <= px < W_left and 0 <= py < H_left:
    #         pixel = bev_left[py, px]
    #
    #         # === 1. 如果想单独将 "实线/虚线/背景" 其中之一分别进行后处理,那么就选择性的打开这些中的一个或多个 ===
    #         # if np.all(pixel == [255, 0, 255]):    # 实线
    #         #     risks_clamped[i] = 1.0
    #         # if np.all(pixel == [255, 140, 255]):  # 虚线
    #         #     risks_clamped[i] = 0.9
    #         if np.all(pixel == [0, 0, 0]):          # 背景
    #             risks_clamped[i] = max(risks_clamped[i], background_risk[py, px])
    #
    #         # === 2. 如果想将 "实线&虚线&背景" 全部进行后处理,那么就单独打开这些 ===
    #         # if not road_mask[py, px]:  # 非可通行道路(背景+车道线)
    #         #     risks_clamped[i] = max(risks_clamped[i], background_risk[py, px])
    #
    # risks = risks_clamped

    if SaveImage:
        # =========================
        # 1) 在 bev (语义 BEV 左半图) 上绘制极点
        # =========================
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(bev_left)
        cmap = plt.get_cmap('RdYlGn_r')  # 红=高风险，绿=低风险
        sc = ax.scatter(poles[:, 0], poles[:, 1], c=risks, cmap=cmap, s=5, vmin=0, vmax=max(risks.max(), 1e-6))  # s 控制极点的半径大小
        ax.axis('off')  # 关闭坐标轴
        plt.colorbar(sc, fraction=0.03, pad=0.02)
        # 存图
        save_dir = os.path.join(os.path.dirname(os.path.dirname(bev_path)), "bev_polarpoint_25x36")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, os.path.basename(bev_path))
        plt.savefig(save_path, dpi=150)
        plt.close()

        # =========================
        # 2) 在 bev_rgb 鸟瞰图上绘制同一批极点（按物理尺度映射）
        # =========================
        bev_rgb_path = bev_path.replace(os.sep + "bev" + os.sep,
                                        os.sep + "bev_rgb" + os.sep)

        if os.path.exists(bev_rgb_path):
            bev_rgb_img = np.array(Image.open(bev_rgb_path))

            # 分辨率
            H_left, W_left, _ = bev_left.shape      # 一般是 192 x 192
            H_rgb,  W_rgb,  _ = bev_rgb_img.shape   # 一般是 1024 x 1024

            # -----------------------------
            # 1) BEV 的几何参数（和 ObsManager 一致）
            # -----------------------------
            PIXELS_PER_METER = 5.0          # self._pixels_per_meter
            BEV_WIDTH = 192                 # self._width
            PIXELS_EV_TO_BOTTOM = 40        # self._pixels_ev_to_bottom

            # ego 在 BEV 图像中的像素位置
            ego_x_bev, ego_y_bev = BEV_WIDTH / 2.0, BEV_WIDTH - PIXELS_EV_TO_BOTTOM  # 96, 152

            # -----------------------------
            # 2) bev_rgb 的几何参数（由相机高度 + FOV 决定）
            # -----------------------------
            h_cam = 70.0       # 相机高度
            fov_deg = 50.0      # 相机 FOV
            R_rgb = h_cam * math.tan(math.radians(fov_deg / 2.0))  # ground 上半宽（米）

            # 每个像素对应的米（近似视为正交投影）
            meters_per_pix_rgb = (2.0 * R_rgb) / W_rgb

            # ego 在 bev_rgb 图像中的像素位置（相机挂在自车正上方，车大致在图像中心）
            ego_x_rgb, ego_y_rgb = W_rgb / 2.0, H_rgb / 2.0

            # -----------------------------
            # 3) BEV 像素坐标 -> 以 ego 为中心的“米”坐标
            # -----------------------------
            # poles: (N, 2)，BEV 中的极点像素坐标
            offset_bev = np.zeros_like(poles, dtype=np.float32)
            offset_bev[:, 0] = poles[:, 0] - ego_x_bev   # 以 ego 为原点的 BEV 像素偏移
            offset_bev[:, 1] = poles[:, 1] - ego_y_bev

            # 把 BEV 像素偏移转成米
            offset_m = offset_bev / PIXELS_PER_METER     # (N, 2)，单位: 米

            # -----------------------------
            # 4) 米坐标 -> bev_rgb 像素坐标
            # -----------------------------
            offset_rgb = offset_m / meters_per_pix_rgb   # (N, 2)，变成在 bev_rgb 中的像素偏移

            poles_rgb = np.zeros_like(poles, dtype=np.float32)
            poles_rgb[:, 0] = ego_x_rgb + offset_rgb[:, 0]
            poles_rgb[:, 1] = ego_y_rgb + offset_rgb[:, 1]

            # -----------------------------
            # 5) 绘制到 bev_rgb 上
            # -----------------------------
            fig = plt.figure(figsize=(W_rgb / 100.0, H_rgb / 100.0), dpi=100)
            ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
            ax.imshow(bev_rgb_img)
            sc = ax.scatter(
                poles_rgb[:, 0], poles_rgb[:, 1],
                c=risks, cmap=cmap, s=5,
                vmin=0, vmax=max(risks.max(), 1e-6)
            )
            ax.axis('off')
            # 不需要 colorbar，就不要加

            bev_rgb_save_dir = os.path.join(
                os.path.dirname(os.path.dirname(bev_rgb_path)),
                "bev_rgb_polarpoint_25x36"
            )
            os.makedirs(bev_rgb_save_dir, exist_ok=True)
            bev_rgb_save_path = os.path.join(
                bev_rgb_save_dir,
                os.path.basename(bev_rgb_path)
            )
            plt.savefig(bev_rgb_save_path, dpi=100)
            plt.close()
        else:
            print(f"[Warn] 对应的 bev_rgb 不存在: {bev_rgb_path}")


    # --- 保存极点标签为 JSON (二维数组，左上对应最远端最左极点) ---
    if SaveJson:
        rows, cols = 25, 36  # 根据生成极点时的设定  baseline
        risks_2d = risks.reshape((rows, cols))  # 原顺序: 最近 → 最远, 左 → 右
        # 将行翻转，使最远层对应二维数组左上角
        risks_2d = np.flipud(risks_2d).tolist() # 现顺序: 最远 → 最近, 左 → 右
        json_save_dir = os.path.join(os.path.dirname(os.path.dirname(bev_path)), "polarpoint_labels_30x36")
        os.makedirs(json_save_dir, exist_ok=True)
        json_save_path = os.path.join(json_save_dir, os.path.basename(bev_path).replace(".png", ".json"))
        with open(json_save_path, 'w') as f:
            json.dump(risks_2d, f)


if __name__ == "__main__":
    if DEBUG:
        file_name = "/home/liulei/ll/PPBEV_ll/data/town01_val/routes_town01_val_12_04_11_10_40/packed_data.npy"  # 只需要修改file_name就行
        data = np.load(file_name, allow_pickle=True).item()

        front_img_path = data["front_img"][index]
        vehicles_info = data['vehicles_info'][index]  # 当前帧
        future_vehicles_info = data['future_vehicles_info'][index]  # 未来4帧，shape: (4, N_t, 8)
        ego_yaw = np.array(math.pi / 2)  # BEV 坐标下自车 yaw

        # === 和非 DEBUG 模式保持一致的时间权重设置 ===
        weight_mode = "exp_decay"  # 或 "manual" / "uniform"
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

        # DEBUG 下也走多帧融合 + 归一化的完整路径
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
            data_paths = glob.glob(os.path.join(root_dir, "routes_*", "packed_data.npy"))  # root_dir根目录下每个子文件夹下的packed_data.npy文件的全局路径

            for data_path in data_paths:
                data = np.load(data_path, allow_pickle=True).item()

                front_imgs = data["front_img"]          # 列表，每个元素对应一帧
                vehicles_infos = data["vehicles_info"]  # 列表，每个元素是车辆信息列表
                ego_yaw = np.array(math.pi/2)           # 每帧自车yaw(BEV坐标系下),这个值是不变的

                for idx, (front_img_path, vehicles_info) in enumerate(
                        tqdm(zip(front_imgs, vehicles_infos), total=len(front_imgs),
                             desc=f" 🐱 {os.path.basename(root_dir)}", leave=False)):  # idx是帧编号

                    future_vehicles_info = data['future_vehicles_info'][idx]  # shape: (4, N_t, 8) 未来4帧的周围车/行人信息

                    # === 选择模式: "manual" / "uniform" / "exp_decay" ===
                    weight_mode = "exp_decay"
                    if weight_mode == "manual":      # 手动模式: 完全自己定义每一帧的权重
                        time_weights = get_time_weights(num_frames=1 + len(future_vehicles_info),
                                                        mode="manual",
                                                        manual_weights=[0.5, 0.25, 0.15, 0.1, 0.0])
                    elif weight_mode == "uniform":   # 均匀模式: 所有帧的权重完全相同
                        time_weights = get_time_weights(num_frames=1 + len(future_vehicles_info),
                                                        mode="uniform")
                    elif weight_mode == "exp_decay": # 指数衰减模式: 当前帧权重最大，随着时间推移，未来帧权重按指数规律递减
                        time_weights = get_time_weights(num_frames=1 + len(future_vehicles_info),
                                                        mode="exp_decay",
                                                        decay_rate=0.7)

                    if SingleFrame:
                        process_one(front_img_path, vehicles_info, ego_yaw=ego_yaw,
                                    scale=2.0,
                                    r_max=100,
                                    future_vehicles_info =None,
                                    time_weights = None)
                    else:
                        process_one(front_img_path, vehicles_info, ego_yaw=ego_yaw,
                                    scale=2.0, r_max=100,
                                    future_vehicles_info=future_vehicles_info,
                                    time_weights=time_weights)