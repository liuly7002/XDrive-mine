import json
import os
import sys
import tqdm

"""
⚠️⚠️⚠️
修改2处：
    routes_type = ["original"]
    towns = ["town02"]    
⚠️⚠️⚠️
"""


def remove_files(root, index, items={"bev": ".png", "meta": ".json", "rgb": ".png", "supervision": ".npy"}):
    # items = {"measurements":".json", "rgb_front":".png"}
    items = {}
    sub_folders = list(os.listdir(root))
    for sub_folder in sub_folders:
        if len(list(os.listdir(os.path.join(root, sub_folder)))) == 0:
            break
        items[sub_folder] = "." + list(os.listdir(os.path.join(root, sub_folder)))[0].split(".")[-1]
    for k, v in items.items():
        data_folder = os.path.join(root, k)
        total_len = len(os.listdir(data_folder))
        for i in range(index, total_len):
            file_name = str(i).zfill(4) + v
            file_name = os.path.join(data_folder, file_name)
            os.remove(file_name)


if __name__ == '__main__':
    routes_type = ["00"]        # ⚠️ 修改1
    towns = ["kuangshan"]       # ⚠️ 修改2

    result_path = "/home/liulei/ll/XDrive-mine/data"
    result_pattern = "data_collect_routes_{}_{}_results.json"  # town, type

    data_path = "/home/liulei/ll/XDrive-mine/data"
    data_pattern = "{}_{}"  # town, type

    for type in routes_type:
        print(type)
        for town in tqdm.tqdm(towns):
            result_file = result_pattern.format(town, type)   # data_collect_routes_town01_addition_results.json
            data_folder = data_pattern.format(town, type)     # town01_addition
            data_folder = os.path.join(data_path, data_folder)      # /home/liulei/ll/PPBEV_ll/data/town01_addition
            sub_folders = os.listdir(data_folder) 
            sub_folders = sorted(list(sub_folders))

            # read the record of each route
            with open(os.path.join(result_path, result_file), 'r') as f:
                records = json.load(f)
            records = records["_checkpoint"]["records"]
            for index, record in enumerate(records):
                route_data_folder = os.path.join(data_folder, sub_folders[index])
                total_length = len(os.listdir(os.path.join(route_data_folder, "measurements")))
                if record["scores"]["score_composed"] >= 100:
                    continue
                # timeout or blocked, remove the last ones where the vehicle stops
                if len(record["infractions"]["route_timeout"]) > 0 or \
                        len(record["infractions"]["vehicle_blocked"]) > 0:
                    stop_index = 0
                    for i in range(total_length - 1, 0, -1):
                        with open(os.path.join(route_data_folder, "measurements", str(i).zfill(4)) + ".json",
                                  'r') as mf:
                            speed = json.load(mf)["speed"]
                            if speed > 0.1:
                                stop_index = i
                                break
                    stop_index = min(total_length, stop_index + 20)
                    remove_files(route_data_folder, stop_index)
                # collision or red-light
                elif len(record["infractions"]["red_light"]) > 0 or \
                        len(record["infractions"]["collisions_pedestrian"]) > 0 or \
                        len(record["infractions"]["collisions_vehicle"]) > 0 or \
                        len(record["infractions"]["collisions_layout"]) > 0:
                    stop_index = max(0, total_length - 10)
                    remove_files(route_data_folder, stop_index)