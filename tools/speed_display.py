import os
import json
import glob
import matplotlib.pyplot as plt


def plot_speed_from_json_folder(folder_path, save_path=None):
    """
    读取 folder_path 下所有 json 文件中的 speed 字段，并绘制曲线
    :param folder_path: json 文件夹路径
    :param save_path: 若不为 None，则将图像保存到该路径
    """
    json_files = sorted(glob.glob(os.path.join(folder_path, "*.json")))

    if len(json_files) == 0:
        print(f"在文件夹 {folder_path} 下没有找到 .json 文件")
        return

    speeds = []
    valid_files = []

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "speed" in data:
                speeds.append(data["speed"])
                valid_files.append(os.path.basename(json_file))
            else:
                print(f"文件中没有 speed 字段，已跳过: {json_file}")

        except Exception as e:
            print(f"读取失败 {json_file}，错误: {e}")

    if len(speeds) == 0:
        print("没有成功读取到任何 speed 数据")
        return

    x = list(range(len(speeds)))

    plt.figure(figsize=(12, 6))
    plt.plot(x, speeds, marker='o')
    plt.xlabel("Frame Index")
    plt.ylabel("Speed")
    plt.title("Speed Curve")
    plt.grid(True)

    # 如果点数不多，可以把文件名显示在横轴
    if len(valid_files) <= 30:
        plt.xticks(x, valid_files, rotation=45, ha='right')

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"图像已保存到: {save_path}")

    plt.show()


if __name__ == "__main__":
    folder_path = "/home/liulei/ll/XDrive-mine/data/kuangshan_00/11_04_20_16_48_29/measurements"   # json 文件夹路径
    save_path = "speed_curve.png"               # 不想保存的话可以改成 None

    plot_speed_from_json_folder(folder_path, save_path)