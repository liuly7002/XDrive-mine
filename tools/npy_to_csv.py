import os
import csv
import numpy as np

"""
函数功能：
将.npy文件转换为.csv以便检查.npy文件的内容
"""

SubfolderProcessing = False
# True: 对town_01\town_02...下的每个子文件夹(也就是每条道路)下的packed_data.npy进行处理
# False: 对town_01\town_02...根目录下的packed_data.npy进行处理

def export_npy_to_csv(npy_file):
    data = np.load(npy_file, allow_pickle=True).item()

    # 获取有效长度（列表型字段中最短的）
    list_lengths = [len(v) for v in data.values() if isinstance(v, (list, np.ndarray))]
    if not list_lengths:
        print(f"❌ 没有有效数据: {npy_file}")
        return
    num_rows = min(list_lengths)

    csv_file = npy_file.replace(".npy", ".csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)

        # 写入字段名作为表头
        keys = list(data.keys())
        writer.writerow(keys)

        for i in range(num_rows):
            row = []
            for key in keys:
                value = data[key]
                try:
                    if isinstance(value, (list, np.ndarray)) and len(value) > i:
                        row.append(str(value[i]))  # 当前帧的数据
                    else:
                        row.append(str(value))  # 单值（如静态设置）
                except Exception as e:
                    row.append("ERROR")
                    print(f"⚠️ 数据错误 key={key}, index={i}, 错误: {e}")
            writer.writerow(row)


    print(f"✅ 导出成功: {csv_file}")

if __name__ == "__main__":
    root_dir = "/home/liulei/ll/PPBEV_ll/data/town02_val"   # ⚠️ 修改1

    if SubfolderProcessing:
        for subdir in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, subdir)
            npy_file = os.path.join(folder_path, "packed_data_normal.npy")
            if os.path.isfile(npy_file):
                try:
                    export_npy_to_csv(npy_file)
                except Exception as e:
                    print(f"❌ 错误处理 {npy_file}: {e}")
    else:
        npy_file = os.path.join(root_dir, "packed_data_normal.npy")
        if os.path.isfile(npy_file):
            try:
                export_npy_to_csv(npy_file)
            except Exception as e:
                print(f"❌ 错误处理 {npy_file}: {e}")
