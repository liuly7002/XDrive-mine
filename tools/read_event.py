import os
import csv
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

"""
函数功能：
将log/TCP/lightning_logs/文件夹下所有子文件夹中的事件文件中的"val_ade_normal", "val_fde_normal"等key转换为.csv
"""

def events_to_csv_selected(event_file, csv_file, tags_to_keep):
    """
    将 TensorBoard 事件文件转换为 CSV，只保留指定标签且只记录实际有数值的 step
    """
    ea = EventAccumulator(event_file)
    ea.Reload()
    available_tags = ea.Tags()["scalars"]

    # 过滤只保留指定标签
    tags = [tag for tag in tags_to_keep if tag in available_tags]

    with open(csv_file, "w", newline='') as f:
        writer = csv.writer(f)
        # 表头
        writer.writerow(["step", "tag", "value"])
        # 遍历每个 tag 的实际数据
        for tag in tags:
            scalars = ea.Scalars(tag)
            for s in scalars:
                writer.writerow([s.step, tag, s.value])

    print(f"{event_file} 已转换为 CSV -> {csv_file}")

def batch_convert_events_selected(log_root, output_dir, tags_to_keep):
    """
    批量将所有子文件夹下事件文件转换为 CSV，只保留指定标签
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for version in os.listdir(log_root):
        version_path = os.path.join(log_root, version)
        if not os.path.isdir(version_path):
            continue

        event_files = [f for f in os.listdir(version_path) if f.startswith("events.out.tfevents")]
        for ef in event_files:
            ef_path = os.path.join(version_path, ef)
            csv_file = os.path.join(output_dir, f"{version}_{ef}.csv")
            print(f"Converting {ef_path} -> {csv_file}")
            events_to_csv_selected(ef_path, csv_file, tags_to_keep)

# 使用示例
log_root = "/home/liulei/ll/PPBEV_ll/A_result/A_文章所有实验/迁移实验-nuplan"
output_dir = "/home/liulei/ll/PPBEV_ll/A_result/A_文章所有实验/迁移实验-nuplan/A_result_csv"
tags_to_keep=["val_pp_corr", "val_pp_l1", "val_pp_mse",
              "val_ade_normal", "val_fde_normal", "val_mr_normal", "val_wp_loss",
              "val_action_loss", "val_throttle", "val_steer", "val_brake",
              "future_action_loss", "val_future_throttle", "val_future_steer", "val_future_brake"
              ]  # 修改这里
batch_convert_events_selected(log_root, output_dir, tags_to_keep)
