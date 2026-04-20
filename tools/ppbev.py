# import os
# from PIL import Image, ImageDraw
# import json
# import math
#
# COLOR_BLACK = (0, 0, 0) # background 0
# COLOR_BLUE = (0, 0, 255) # vehicle 1
# COLOR_ALUMINIUM_5 = (255, 255, 255) # road 2
# COLOR_CYAN = (0, 255, 255) # walker 1
#
# COLOR_RED = (255, 0, 0) # tl_red
# COLOR_Lane = (0, 255, 0) # lane
#
# # 标签对应绘制颜色
# LABEL_COLORS = {
#     0: (255, 255, 0),  # 黄色
#     1: (255, 0, 0),    # 红色
#     2: (0, 255, 0)     # 绿色
# }
#
# def classify(r, g, b):
# 	if b == 255 and r == 0 and g == 0:  # vehicle
# 		return 1
# 	if b == 255 and r == 0 and g == 255: # vehicle
# 		return 1
# 	if b == 0 and r == 0 and g == 0: # background
# 		return 0
# 	else:
# 		return 2
#
# root = './data/town01_original/'
# routes = sorted(os.listdir(root))
#
# # Define the center and radius of the fan-shaped region
# center = (96, 152)
# # radius = [19.5*5, 16.5*5, 13.5*5, 10.5*5, 8*5, 6*5, 4.5*5, 3.5*5]
# radius = [19.5*5, 17.5*5, 16*5, 14.5*5, 13*5, 11.5*5, 10*5, 9*5, 8*5, 7*5, 6*5, 5.5*5, 5*5, 4.5*5, 4*5, 3.5*5]
#
# # Define the angles of the fan-shaped region
# start_angle = -140
# end_angle = -40
#
# # Define the number of points to detect along the border of the fan
# num_points = 27  # 15_sparse; 21_light; 27_normal; 33_thick; 41_dense
# counter = 1
#
# # Calculate the angle increment between each point
# angle_increment = (end_angle - start_angle) / (num_points-1)
#
#
# for route in routes:
# 	route_path = os.path.join(root, route)
# 	bev_path = os.path.join(route_path, 'bev')
# 	print(counter)
# 	counter += 1
# 	graph_root = os.path.join(route_path, 'graph_gt')
# 	os.makedirs(graph_root, exist_ok=True)
# 	bevs = sorted(os.listdir(bev_path))
#
# 	# bev_ppbev_dir = os.path.join(route_path, 'bev_ppbev')
# 	# os.makedirs(bev_ppbev_dir, exist_ok=True)
#
# 	for img in bevs:
# 		label_image = []
# 		img_path = os.path.join(bev_path, img)
# 		graph_name = img[:-4] + '.json'
# 		graph_path = os.path.join(graph_root, graph_name)
#
# 		# img = Image.open(img_path)
# 		# rgb_im = img.convert('RGB')
# 		# draw = ImageDraw.Draw(rgb_im)
# 		img_name = img  # 原文件名
# 		img_pil = Image.open(img_path)
# 		rgb_im = img_pil.convert('RGB')
# 		draw = ImageDraw.Draw(rgb_im)
#
# 		for rad in radius:
# 			label_layer = []
# 			for i in range(num_points):
# 				# Calculate the angle of the current point
# 				angle = start_angle + i * angle_increment
#
# 				# Calculate the coordinates of the current point
# 				x = center[0] + rad * math.cos(math.radians(angle))
# 				y = center[1] + rad * math.sin(math.radians(angle))
#
# 				# Obtain the RGB value of the current point in the original image
# 				# r, g, b = img.getpixel((x, y))
# 				r, g, b = rgb_im.getpixel((x, y))
#
# 				# label_layer.append(img.getpixel((x, y)))
# 				label = classify(r, g, b)
# 				label_layer.append(label)
#
# 				# ✅ 新增：绘制极点
# 				if label in LABEL_COLORS:
# 					color = LABEL_COLORS[label]
# 					draw.ellipse(
# 						[(x - 1.5, y - 1.5), (x + 1.5, y + 1.5)],
# 						fill=color
# 					)
#
# 			label_image.append(label_layer)
#
# 		with open(graph_path, 'w') as jfile:
# 			json.dump(label_image, jfile)
#
# 		# ✅ 保存绘制后的可视化图像
# 		# save_path = os.path.join(bev_ppbev_dir, img_name)
# 		# rgb_im.save(save_path)






import os
import numpy as np

# 原始 npy 文件路径
original_npy_path = "./data/town01_original/packed_data.npy"
# 新生成的 npy 文件路径
new_npy_path = "./data/town01_original/packed_data_normal.npy"

# 加载原始数据
data = np.load(original_npy_path, allow_pickle=True).item()

# 新增 'graph' 键
if "polarpoint_labels" in data:
    graph_paths = []
    for seq in data["polarpoint_labels"]:  # 每个 seq 是多帧 list
        graph_seq = [p.replace("/polarpoint_labels/", "/graph_gt/") for p in seq]
        graph_paths.append(graph_seq)
    data["graph"] = graph_paths
    print("✅ Added 'graph' key successfully.")
else:
    print("⚠️ 'polarpoint_labels' key not found in the npy file.")

# 保存为新的 npy 文件，不覆盖原文件
np.save(new_npy_path, data, allow_pickle=True)
print(f"🐱 Saved updated packed_data to {new_npy_path}")

