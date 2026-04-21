import os
import json
import numpy as np
import tqdm

"""
⚠️⚠️⚠️
修改1处：
    # 本地
    data_path = "/home/liulei/ll/PPBEV_ll/data/town01_original"
    # 服务器
    data_path = "/home/kemove/ll/PPBEV_ll/data/town01_original"
⚠️⚠️⚠️
"""


# 本地
data_path = "/home/liulei/ll/XDrive-mine/data/kuangshan_00"  # ⚠️ 修改: CARLA 数据集的根目录，每个 route 在该路径下一个子文件夹中。
# 服务器
# data_path = "/home/kemove/ll/PPBEV_ll/data/town01_original"  # ⚠️ 修改: CARLA 数据集的根目录，每个 route 在该路径下一个子文件夹中。

INPUT_FRAMES = 1   # 网络的输入帧数(历史帧,也就是当前1帧)
FUTURE_FRAMES = 4  # 网络的预测帧数(未来帧,也就是不包括当前1帧的未来连续4帧)

def gen_single_route(route_folder):
    length = len(os.listdir(os.path.join(route_folder, 'measurements')))  # measurements中的文件的数目,每个文件与一帧图像对应，表示了当前图像下的信息
    if length < INPUT_FRAMES + FUTURE_FRAMES:  # 如果帧数太少(不足以生成一个有效序列), 就直接返回
        return

    # 未来帧信息(本文使用了4帧)
    seq_future_x, seq_future_y, seq_future_theta = [], [], []
    seq_future_feature, seq_future_action = [], []
    seq_future_action_mu, seq_future_action_sigma = [], []
    seq_future_only_ap_brake = []

    # 当前帧信息
    seq_input_x, seq_input_y, seq_input_theta = [], [], []
    seq_front_img, seq_feature, seq_value, seq_speed = [], [], [], []
    seq_action, seq_action_mu, seq_action_sigma = [], [], []
    seq_x_target, seq_y_target, seq_target_command = [], [], []
    seq_only_ap_brake = []

    # 全部信息追加进list
    full_seq_x, full_seq_y, full_seq_theta = [], [], []
    full_seq_feature, full_seq_action = [], []
    full_seq_action_mu, full_seq_action_sigma = [], []
    full_seq_only_ap_brake = []

    # 全部信息追加进list
    full_ev_loc_info_world = []       # 自车当前帧的世界坐标系位置和航向,形式为[x,y,yaw]
    full_vehicles_info = []           # 周围感兴趣区域内车辆的信息
    full_walkers_info = []            # 周围感兴趣区域内行人的信息
    # 未来帧信息(本文使用了4帧)
    seq_future_ev_loc_info_world = []    # 自车未来4帧的世界坐标系位置和航向,形式为[[x,y,yaw],[x,y,yaw],[x,y,yaw],[x,y,yaw]]
    seq_future_vehicles_info = []        # 周围感兴趣区域内车辆的信息[[],[],[],[]]
    seq_future_walkers_info = []         # 周围感兴趣区域内行人的信息[[],[],[],[]]

    # 全部信息追加进list
    full_route_pixels = []            # 全局路径的靠近自车前20个点在BEV中的像素坐标
    full_route_world = []             # 全局路径的靠近自车前20个点的carla世界坐标
    full_route_relative_yaw_rad = []  # 全局路径的靠近自车前20个点相对于自车的航向角,负值代表向左,正值代表向右
    # 未来帧信息(本文使用了4帧)
    seq_route_pixels = []             # 全局路径的靠近自车前20个点在BEV中的像素坐标
    seq_route_world = []              # 全局路径的靠近自车前20个点的carla世界坐标
    seq_route_relative_yaw_rad = []   # 全局路径的靠近自车前20个点相对于自车的航向角,负值代表向左,正值代表向右

    # 1. 开始追加全部信息
    for i in range(length):

        # 1.1 首先处理 measurements 文件夹下的.json文件
        with open(os.path.join(route_folder, "measurements", f"{str(i).zfill(4)}.json"), "r") as read_file:
            measurement = json.load(read_file)  # 加载measurements文件夹中的每一个.json文件

        # GPS坐标系是(x向北y向东) CARLA世界坐标系是(x向东y向北) 这里创建了新的坐标系-(x向东y向北)
        full_seq_x.append(measurement['y'])  # 新坐标系的x坐标
        full_seq_y.append(measurement['x'])  # 新坐标系的y坐标
        full_seq_theta.append(measurement['theta'])  # 自车在使用IMU获取的航向角

        vehicles = measurement.get("vehicles_info", [])  # 获取全部的周围车辆信息

        if vehicles:  # 如果有周围车辆
            ev_loc = vehicles[0].get("ev_loc_info_world", [])  # 自车的carla世界坐标系下的位姿信息[x,y,yaw],因为每个vehicles都有"ev_loc_info_world",所以这里只选择其中一个的就行
        else:
            ev_loc = []  # ✅ 空时设为 []

        full_ev_loc_info_world.append(ev_loc)

        frame_vehicles = []
        for veh in vehicles:  # 遍历每一辆周围车
            if veh.get("type") == "ego_only":
                continue  # ❌ 跳过占位符，自车信息我们只记录在 ev_loc
            bev_x, bev_y = veh.get("bev_position", [0.0, 0.0])  # 在BEV图像中的像素坐标
            yaw = veh.get("yaw", 0.0)                           # 在carla世界坐标系中的航向角
            bev_yaw = veh.get("bev_yaw", 0.0)                   # 在BEV坐标系中的航向角
            world_length = veh.get("world_length", 0.0)         # 在carla世界坐标系中的车长,m
            world_width = veh.get("world_width", 0.0)           # 在carla世界坐标系中的车宽,m
            length_pix = veh.get("length_in_pix", 0.0)          # 在BEV中车长所占用的像素个数,pix
            width_pix = veh.get("width_in_pix", 0.0)            # 在BEV中车宽所占用的像素个数,pix

            frame_vehicles.append([  # 将每一辆周围车的信息先打包成一个list然后追加到frame_vehicles里面
                bev_x, bev_y, yaw, bev_yaw, world_length, world_width, length_pix, width_pix
            ])
        full_vehicles_info.append(frame_vehicles)  # ✅ 即使为空帧，也 append([])

        walkers = measurement.get("walker_info", [])     # 获取全部的周围行人信息

        frame_walkers = []
        for walker in walkers:
            if walker.get("type") == "ego_only":
                continue  # ❌ 跳过占位符，自车信息我们只记录在 ev_loc
            bev_x, bev_y = walker.get("bev_position", [0.0, 0.0])
            yaw = walker.get("yaw", 0.0)
            bev_yaw = veh.get("bev_yaw", 0.0)
            world_length = walker.get("world_length", 0.0)
            world_width = walker.get("world_width", 0.0)
            length_pix = walker.get("length_in_pix", 0.0)
            width_pix = walker.get("width_in_pix", 0.0)

            frame_walkers.append([
                bev_x, bev_y, yaw, bev_yaw, world_length, world_width, length_pix, width_pix
            ])
        full_walkers_info.append(frame_walkers)  # ✅ 即使为空也 append([])

        # 全局路径相关
        full_route_pixels.append(measurement['route_pixels'])
        # full_route_world.append(measurement['route_world'])                      # 这是在carla世界坐标系下的坐标
        full_route_world.append([[x, -y] for x, y in measurement['route_world']])  # 转换到新坐标系下
        full_route_relative_yaw_rad.append(measurement['route_relative_yaw_rad'])

        # 1.2 处理 supervision 文件夹下的.npy文件
        roach_supervision_data = np.load(
            os.path.join(route_folder, "supervision", f"{str(i).zfill(4)}.npy"),
            allow_pickle=True
        ).item()  # 加载supervision文件夹中的每一个.npy文件

        full_seq_feature.append(roach_supervision_data['features'])
        full_seq_action.append(roach_supervision_data['action'])            # 当前帧
        full_seq_action_mu.append(roach_supervision_data['action_mu'])
        full_seq_action_sigma.append(roach_supervision_data['action_sigma'])
        full_seq_only_ap_brake.append(roach_supervision_data['only_ap_brake'])

    # 2. 开始追加未来帧信息
    for i in range(INPUT_FRAMES-1, length-FUTURE_FRAMES):  # 0,总文件-4
        with open(os.path.join(route_folder, "measurements", f"{str(i).zfill(4)}.json"), "r") as read_file:
            measurement = json.load(read_file)  # 加载measuremnets文件夹中的每一个.json文件

        # 当前帧
        seq_input_x.append(full_seq_x[i-(INPUT_FRAMES-1):i+1])          # 当前自车的在新坐标系中的x
        seq_input_y.append(full_seq_y[i-(INPUT_FRAMES-1):i+1])          # 当前自车的在新坐标系中的y
        seq_input_theta.append(full_seq_theta[i-(INPUT_FRAMES-1):i+1])  # 当前自车的在新坐标系中的航向角

        # 当前帧后的连续4帧
        seq_future_x.append(full_seq_x[i+1:i+FUTURE_FRAMES+1])          # 当前帧后的连续4帧自车的新局坐标系中的x
        seq_future_y.append(full_seq_y[i+1:i+FUTURE_FRAMES+1])          # 当前帧后的连续4帧自车的新坐标系中的y
        seq_future_theta.append(full_seq_theta[i+1:i+FUTURE_FRAMES+1])  # 当前帧后的连续4帧自车的新坐标系中的航向角

        # 当前帧后的连续4帧
        seq_future_feature.append(full_seq_feature[i+1:i+FUTURE_FRAMES+1])
        seq_future_action.append(full_seq_action[i+1:i+FUTURE_FRAMES+1])
        seq_future_action_mu.append(full_seq_action_mu[i+1:i+FUTURE_FRAMES+1])
        seq_future_action_sigma.append(full_seq_action_sigma[i+1:i+FUTURE_FRAMES+1])
        seq_future_only_ap_brake.append(full_seq_only_ap_brake[i+1:i+FUTURE_FRAMES+1])

        # 当前帧后的连续4帧
        seq_future_ev_loc_info_world.append(full_ev_loc_info_world[i+1:i+FUTURE_FRAMES+1])
        seq_future_vehicles_info.append(full_vehicles_info[i+1:i+FUTURE_FRAMES+1])
        seq_future_walkers_info.append(full_walkers_info[i+1:i+FUTURE_FRAMES+1])

        # 当前帧
        seq_route_pixels.append(full_route_pixels[i-(INPUT_FRAMES-1):i+1])
        seq_route_world.append(full_route_world[i-(INPUT_FRAMES-1):i+1])
        seq_route_relative_yaw_rad.append(full_route_relative_yaw_rad[i-(INPUT_FRAMES-1):i+1])

        roach_supervision_data = np.load(
            os.path.join(route_folder, "supervision", f"{str(i).zfill(4)}.npy"),
            allow_pickle=True
        ).item()

        seq_feature.append(roach_supervision_data["features"])
        seq_value.append(roach_supervision_data["value"])

        # front_img_list = [route_folder.replace(data_path, '') + "/rgb/" + f"{str(i-_).zfill(4)}.png" for _ in range(INPUT_FRAMES-1, -1, -1)]
        front_img_list = [
            "/" + os.path.join(os.path.basename(data_path),
                               os.path.relpath(route_folder, data_path),
                               "rgb",
                               f"{str(i - _).zfill(4)}.png")
            for _ in range(INPUT_FRAMES - 1, -1, -1)
        ]

        seq_front_img.append(front_img_list)

        seq_speed.append(measurement["speed"])
        seq_action.append(roach_supervision_data["action"])
        seq_action_mu.append(roach_supervision_data["action_mu"])
        seq_action_sigma.append(roach_supervision_data["action_sigma"])
        seq_x_target.append(measurement["y_target"])
        seq_y_target.append(measurement["x_target"])
        seq_target_command.append(measurement["target_command"])
        seq_only_ap_brake.append(roach_supervision_data["only_ap_brake"])

    return (
        seq_future_x, seq_future_y, seq_future_theta,
        seq_future_feature, seq_future_action, seq_future_action_mu, seq_future_action_sigma, seq_future_only_ap_brake,
        seq_input_x, seq_input_y, seq_input_theta,
        seq_front_img, seq_feature, seq_value, seq_speed,
        seq_action, seq_action_mu, seq_action_sigma,
        seq_x_target, seq_y_target, seq_target_command, seq_only_ap_brake,
        full_ev_loc_info_world, full_vehicles_info, full_walkers_info,
        seq_future_ev_loc_info_world, seq_future_vehicles_info, seq_future_walkers_info,
        seq_route_pixels, seq_route_world, seq_route_relative_yaw_rad
    )

def gen_sub_folder(folder_path):
    route_list = [folder_path]

    total_future_x, total_future_y, total_future_theta = [], [], []
    total_future_feature, total_future_action = [], []
    total_future_action_mu, total_future_action_sigma = [], []
    total_future_only_ap_brake = []
    total_input_x, total_input_y, total_input_theta = [], [], []
    total_front_img, total_feature, total_value, total_speed = [], [], [], []
    total_action, total_action_mu, total_action_sigma = [], [], []
    total_x_target, total_y_target, total_target_command = [], [], []
    total_only_ap_brake = []

    total_ev_loc_info_world = []
    total_vehicles_info = []
    total_walkers_info = []
    total_future_ev_loc_info_world = []
    total_future_vehicles_info = []
    total_future_walkers_info = []

    total_route_pixels = []
    total_route_world  = []
    total_route_relative_yaw_rad = []

    for route in route_list:
        seq_data = gen_single_route(route)
        if not seq_data:
            continue

        (
            seq_future_x, seq_future_y, seq_future_theta,
            seq_future_feature, seq_future_action, seq_future_action_mu, seq_future_action_sigma, seq_future_only_ap_brake,
            seq_input_x, seq_input_y, seq_input_theta,
            seq_front_img, seq_feature, seq_value, seq_speed,
            seq_action, seq_action_mu, seq_action_sigma,
            seq_x_target, seq_y_target, seq_target_command, seq_only_ap_brake,
            ev_loc_info_world, vehicles_info, walkers_info,
            seq_future_ev_loc_info_world, seq_future_vehicles_info, seq_future_walkers_info,
            seq_route_pixels, seq_route_world, seq_route_relative_yaw_rad
        ) = seq_data

        total_future_x.extend(seq_future_x)
        total_future_y.extend(seq_future_y)
        total_future_theta.extend(seq_future_theta)
        total_future_feature.extend(seq_future_feature)
        total_future_action.extend(seq_future_action)
        total_future_action_mu.extend(seq_future_action_mu)
        total_future_action_sigma.extend(seq_future_action_sigma)
        total_future_only_ap_brake.extend(seq_future_only_ap_brake)
        total_input_x.extend(seq_input_x)
        total_input_y.extend(seq_input_y)
        total_input_theta.extend(seq_input_theta)
        total_front_img.extend(seq_front_img)
        total_feature.extend(seq_feature)
        total_value.extend(seq_value)
        total_speed.extend(seq_speed)
        total_action.extend(seq_action)
        total_action_mu.extend(seq_action_mu)
        total_action_sigma.extend(seq_action_sigma)
        total_x_target.extend(seq_x_target)
        total_y_target.extend(seq_y_target)
        total_target_command.extend(seq_target_command)
        total_only_ap_brake.extend(seq_only_ap_brake)
        total_ev_loc_info_world.extend(ev_loc_info_world)
        total_vehicles_info.extend(vehicles_info)
        total_walkers_info.extend(walkers_info)
        total_future_ev_loc_info_world.extend(seq_future_ev_loc_info_world)
        total_future_vehicles_info.extend(seq_future_vehicles_info)
        total_future_walkers_info.extend(seq_future_walkers_info)
        total_route_pixels.extend(seq_route_pixels)
        total_route_world.extend(seq_route_world)
        total_route_relative_yaw_rad.extend(seq_route_relative_yaw_rad)


    data_dict = {
        'future_x': total_future_x,
        'future_y': total_future_y,
        'future_theta': total_future_theta,
        'future_feature': total_future_feature,
        'future_action': total_future_action,
        'future_action_mu': total_future_action_mu,
        'future_action_sigma': total_future_action_sigma,
        'future_only_ap_brake': total_future_only_ap_brake,
        'input_x': total_input_x,
        'input_y': total_input_y,
        'input_theta': total_input_theta,
        'front_img': total_front_img,
        'feature': total_feature,
        'value': total_value,
        'speed': total_speed,
        'action': total_action,
        'action_mu': total_action_mu,
        'action_sigma': total_action_sigma,
        'x_target': total_x_target,
        'y_target': total_y_target,
        'target_command': total_target_command,
        'only_ap_brake': total_only_ap_brake,
        'ev_loc_info_world': total_ev_loc_info_world,    # [world_x, world_y, yaw]
        'vehicles_info': total_vehicles_info,            # [[bev_x, bev_y, yaw, world_length, world_width, length_pix, width_pix], ...]
        'walkers_info': total_walkers_info,              # [[bev_x, bev_y, yaw, world_length, world_width, length_pix, width_pix], ...]
        'future_ev_loc_info_world': total_future_ev_loc_info_world,    # [world_x, world_y, yaw]
        'future_vehicles_info': total_future_vehicles_info,            # [[bev_x, bev_y, yaw, world_length, world_width, length_pix, width_pix], ...]
        'future_walkers_info': total_future_walkers_info,              # [[bev_x, bev_y, yaw, world_length, world_width, length_pix, width_pix], ...]
        'route_pixels': total_route_pixels,                      # 全局路径的靠近自车前20个点在BEV中的像素坐标
        'route_world': total_route_world,                        # 全局路径的靠近自车前20个点的carla世界坐标
        'route_relative_yaw_rad': total_route_relative_yaw_rad   # 全局路径的靠近自车前20个点相对于自车的航向角,负值代表向左,正值代表向右
    }

    file_path = os.path.join(folder_path, "packed_data.npy")
    np.save(file_path, data_dict)
    return len(total_future_x)

if __name__ == '__main__':

    """
    1. 先生成每个子文件夹(也就是每条route)下的packed_data.npy
    """
    # route_folders = [f for f in os.listdir(data_path) if f.startswith("routes_")]
    route_folders = sorted([f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))])
    # print("Found route folders:", route_folders)

    route_folders = sorted(route_folders)

    total = 0
    for folder in tqdm.tqdm(route_folders):
        route_folder_path = os.path.join(data_path, folder)
        number = gen_sub_folder(route_folder_path)
        total += number

    print(f"Total training samples: {total}")

    """
    2. 将root_dir的每一个子文件夹根目录下的packed_data.npy拼接在一起并保存到root_dir根目录下
    """

    output_file = os.path.join(data_path, "packed_data.npy")

    subfolders = sorted([os.path.join(data_path, d) for d in os.listdir(data_path)
                         if os.path.isdir(os.path.join(data_path, d))])

    merged_data = None

    # for sub in subfolders:
    for sub in tqdm.tqdm(subfolders, desc="Merging packed_data", unit="folder"):
        npy_file = os.path.join(sub, "packed_data.npy")
        if os.path.exists(npy_file):
            # print(f"Loading {npy_file} ...")
            obj = np.load(npy_file, allow_pickle=True).item()  # 取出 dict

            if merged_data is None:
                # 初始化一个 dict，所有 key 对应空 list
                merged_data = {k: [] for k in obj.keys()}

            # 把每个子文件夹的数据扩展到 merged_data 里
            for k, v in obj.items():
                merged_data[k].extend(v)  # 注意这里用 extend，而不是 append
        else:
            print(f"⚠️ Warning: {npy_file} not found, skipped.")

    """
    3. 生成 polarpoint_labels 路径
    """
    if merged_data is not None:
        # 生成 polarpoint_labels 路径
        if "front_img" in merged_data:
            polarpoint_labels = []
            for seq in merged_data["front_img"]:  # front_img 里面是多帧 list
                polar_seq = [
                    p.replace("/rgb/", "/polarpoint_labels_32x32/").replace(".png", ".json")
                    for p in seq
                ]
                polarpoint_labels.append(polar_seq)
            merged_data["polarpoint_labels"] = polarpoint_labels

        # 保存为和原始格式一致：0D ndarray 包含一个 dict
        np.save(output_file, merged_data, allow_pickle=True)
        print(f"🐱 Saved merged packed_data.npy to {output_file}")
    else:
        print("❌ No packed_data.npy files found to merge.")
