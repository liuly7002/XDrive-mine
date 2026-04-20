import numpy as np
import carla
from gym import spaces
import cv2 as cv
from collections import deque
from pathlib import Path
import h5py

from roach.utils.traffic_light import TrafficLightHandler

	
import math
import cv2

COLOR_BLACK = (0, 0, 0)
COLOR_RED = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (0, 0, 255)
COLOR_CYAN = (0, 255, 255)
COLOR_MAGENTA = (255, 0, 255)
COLOR_MAGENTA_2 = (255, 140, 255)
COLOR_YELLOW = (255, 255, 0)
COLOR_YELLOW_2 = (160, 160, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_ALUMINIUM_0 = (238, 238, 236)
COLOR_ALUMINIUM_3 = (136, 138, 133)
COLOR_ALUMINIUM_5 = (46, 52, 54)


def tint(color, factor):
	r, g, b = color
	r = int(r + (255-r) * factor)
	g = int(g + (255-g) * factor)
	b = int(b + (255-b) * factor)
	r = min(r, 255)
	g = min(g, 255)
	b = min(b, 255)
	return (r, g, b)


class ObsManager():
	def __init__(self, obs_configs, criteria_stop=None):
		self._width = int(obs_configs['width_in_pixels'])               # BEV的尺寸 192 单位: 像素
		self._pixels_ev_to_bottom = obs_configs['pixels_ev_to_bottom']  # 40 自车mask中心点在BEV图像的的位置距离BEV图像底部的像素个数
		self._pixels_per_meter = obs_configs['pixels_per_meter']        # 2 2像素/米
		self._history_idx = obs_configs['history_idx']
		self._scale_bbox = obs_configs.get('scale_bbox', True)          # true 是否将b_box放大
		self._scale_mask_col = obs_configs.get('scale_mask_col', 1.1)

		self._history_queue = deque(maxlen=20)         # 存放历史mask 包括周围感兴趣区域内的车辆、行人,绿灯、黄灯、红灯、停止线

		self._image_channels = 3
		self._masks_channels = 3 + 3*len(self._history_idx)
		self._parent_actor = None
		self._world = None

		self._map_dir = Path(__file__).resolve().parent / 'maps'  # 高精地图文件所在目录

		self._criteria_stop =criteria_stop

		super(ObsManager, self).__init__()

	def attach_ego_vehicle(self, ego_vehicle):
		self._parent_actor = ego_vehicle
		self._world = self._parent_actor.get_world()

		# 从高精地图获取信息
		maps_h5_path = self._map_dir / (self._world.get_map().name + '.h5')
		with h5py.File(maps_h5_path, 'r', libver='latest', swmr=True) as hf:
			self._road = np.array(hf['road'], dtype=np.uint8)  # 从高精地图上读取道路 0 1 mask
			self._lane_marking_all = np.array(hf['lane_marking_all'], dtype=np.uint8)
			self._lane_marking_white_broken = np.array(hf['lane_marking_white_broken'], dtype=np.uint8)  # 白色虚线

			# 读取全局坐标偏移，单位是米，用于将地图坐标与世界坐标对齐；
			self._world_offset = np.array(hf.attrs['world_offset_in_meters'], dtype=np.float32)
			# 校验地图像素与实际米数的比例关系，保证坐标转换的一致性。
			assert np.isclose(self._pixels_per_meter, float(hf.attrs['pixels_per_meter']))

		self._distance_threshold = np.ceil(self._width / self._pixels_per_meter)  # 192/2=96m  向上取整

	@staticmethod
	def _get_stops(criteria_stop):
		"""
		这个函数通常用于自动驾驶评测框架中（比如 Carla 的 Leaderboard）：
		1.确认当前是否有停车任务需要完成。
		2.获取该停车任务对应的物理信息(位置 + 范围)。
		3.用于判断车辆是否进入停车区域、是否完成停车等。

		return:
		1.如果没有满足条件的停车标志，返回空列表 []
		2.否则返回一个列表，包含停车点的详细信息(用于后续路径规划或停车行为判断)
		"""
		stop_sign = criteria_stop._target_stop_sign
		stops = []
		if (stop_sign is not None) and (not criteria_stop._stop_completed):
			bb_loc = carla.Location(stop_sign.trigger_volume.location)
			bb_ext = carla.Vector3D(stop_sign.trigger_volume.extent)
			bb_ext.x = max(bb_ext.x, bb_ext.y)
			bb_ext.y = max(bb_ext.x, bb_ext.y)
			trans = stop_sign.get_transform()
			stops = [(carla.Transform(trans.location, trans.rotation), bb_loc, bb_ext)]
		return stops

	def draw_actor_centers_bev(self, image, actors, M_warp, ev_loc, ev_rot_yaw_rad, color=(0, 255, 255), radius=4):
		"""
		在 BEV（Bird's Eye View）图像上绘制每个 actor 的中心点和朝向箭头，并返回其关键属性信息。

		绘图说明：
		- 用圆点表示 actor 的中心位置（车辆、行人等）
		- 用箭头表示其朝向方向（从中心朝前 2 米）

		返回每个 actor 的以下信息（以字典形式组织）：
		- 'world_position': (x, y)，actor 在世界坐标系中的位置（单位：米）
		- 'bev_position': (x, y)，actor 在 BEV 图像中的像素位置（单位：像素）
		- 'yaw': 朝向角，世界坐标系下的偏航角（单位：度）
		- 'world_length': 实际长度（单位：米）
		- 'world_width': 实际宽度（单位：米）
		- 'length_in_pix': 在 BEV 图像中的长度（单位：像素）
		- 'width_in_pix': 在 BEV 图像中的宽度（单位：像素）

		参数说明：
		- image: BEV 图像（numpy 数组）
		- actors: actor 列表，每个元素是一个三元组：(Transform, _, bb_ext)
		- M_warp: 世界坐标到 BEV 图像坐标的仿射变换矩阵
		- color: 绘图颜色（BGR 格式）
		- radius: 中心点圆的半径（单位：像素）
		"""

		# 存储每个 actor 的世界坐标和图像坐标
		actor_positions = []

		for transform, _, bb_ext in actors:
			x = transform.location.x                  # carla世界坐标系下的x 单位:m
			y = transform.location.y                  # carla世界坐标系下的y 单位:m
			yaw = np.radians(transform.rotation.yaw)  # carla世界坐标系下的航向角yaw 弧度

			length = (bb_ext.x * 2)  # 车辆或行人的长 单位:m
			width = (bb_ext.y * 2)   # 车辆或行人的宽 单位:m
			length_in_pix = (bb_ext.x * 2 * self._pixels_per_meter)   # 车辆或行人的长 单位:像素
			width_in_pix = (bb_ext.y * 2 * self._pixels_per_meter)    # 车辆或行人的宽 单位:像素

			# 世界坐标 → 图像像素坐标（先转像素坐标，再做仿射变换）
			world_pt = np.array([[self._world_to_pixel(transform.location)]], dtype=np.float32)  # shape: (1,1,2)
			bev_pt = cv2.transform(world_pt, M_warp)[0, 0]  # shape: (2,)
			center_x, center_y = int(bev_pt[0]), int(bev_pt[1])  # 周围车辆在BEV中的像素坐标[x,y]

			# 朝向向量在世界坐标系中：前向2米
			forward_vec = np.array([x + 2.0 * math.cos(yaw), y + 2.0 * math.sin(yaw)])
			forward_pt = np.array([[self._world_to_pixel(carla.Location(x=forward_vec[0], y=forward_vec[1], z=0))]], dtype=np.float32)
			bev_forward_pt = cv2.transform(forward_pt, M_warp)[0, 0]
			fx, fy = int(bev_forward_pt[0]), int(bev_forward_pt[1])  # 方向箭头顶点在BEV中的像素坐标[x,y]

			yaw_bev = np.arctan2(fy - center_y, fx - center_x)# 周围感兴趣车辆或行人 yaw 朝向(BEV坐标系下的角度)

			# ✅ 过滤掉超出 BEV 图像范围的对象
			if not (0 <= center_x < 192 and 0 <= center_y < 192 and
					0 <= fx < 192 and 0 <= fy < 192):
				continue

			# 画圆点和箭头
			# cv2.circle(image, (center_x, center_y), radius, color, -1)
			# cv2.arrowedLine(image, (center_x, center_y), (fx, fy), color, 1, tipLength=0.4)

			# 记录每个 actor 的世界坐标、图像坐标和朝向、长、宽
			actor_positions.append({
				'ev_loc_info_world':(ev_loc.x, ev_loc.y, ev_rot_yaw_rad),  # 自车的世界坐标系位置[x,y]和航向角yaw
				'world_position': (x, y),              # 周围感兴趣车辆或行人在carla世界坐标系中的位置
				'bev_position': (center_x, center_y),  # 周围感兴趣车辆或行人在 BEV 图像坐标位置
				'yaw': yaw,                            # 周围感兴趣车辆或行人 yaw 朝向(世界坐标系下的角度)
				'bev_yaw': yaw_bev,                    # 周围感兴趣车辆或行人 yaw 朝向(BEV坐标系下的角度)
				'world_length': length,                # 车辆或行人的长 单位:m
				'world_width': width,                  # 车辆或行人的宽 单位:m
				'length_in_pix': length_in_pix,        # 车辆或行人的长 单位:像素
				'width_in_pix': width_in_pix,          # 车辆或行人的宽 单位:像素
				'type': 'not_ego_only'                 # 明确标记是自车占位信息
			})

		if len(actor_positions) == 0:
			actor_positions.append({
				'ev_loc_info_world': (ev_loc.x, ev_loc.y, ev_rot_yaw_rad),
				'world_position': (-1, -1),
				'bev_position': (-1, -1),
				'yaw': -1,
				'bev_yaw': -1,
				'world_length': -1,
				'world_width': -1,
				'length_in_pix': -1,
				'width_in_pix': -1,
				'type': 'ego_only'  # 明确标记是自车占位信息
			})

		# 返回记录的所有 actor 的信息
		return actor_positions


	# 关键函数 有关BEV的处理都在这里
	def get_observation(self, route_plan):
		# 自车状态信息获取
		ev_transform = self._parent_actor.get_transform()
		ev_loc = ev_transform.location                            # 自车carla世界坐标系下的位置(中心坐标[x,y,z])
		ev_rot = ev_transform.rotation                            # 自车carla世界坐标系下的姿态(欧拉角yaw pitch roll)
		ev_rot_yaw_rad = math.radians(ev_transform.rotation.yaw)  # 自车carla世界坐标系下的航向角yaw,弧度
		ev_bbox = self._parent_actor.bounding_box                 # 自车尺寸 一般是[l/2, w/2, h/2]

		# 判断周围车辆/行人是否在感兴趣区域内
		# x方向在39m之内 y方向在39m之内 z方向在8m之内
		def is_within_distance(w):
			c_distance = abs(ev_loc.x - w.location.x) < self._distance_threshold \
				and abs(ev_loc.y - w.location.y) < self._distance_threshold \
				and abs(ev_loc.z - w.location.z) < 8.0  # 空间距离不超过阈值是周围车
			c_ev = abs(ev_loc.x - w.location.x) < 1.0 and abs(ev_loc.y - w.location.y) < 1.0  # 距离过近说明是自车
			return c_distance and (not c_ev)  # 所以返回的是True False, 在感兴趣区域内并且不是自车 也就是该实体是否在附近

		"""	
		# 筛选出在感兴趣区域内的 vehicles 和 walkers
		vehicle_bbox_list = self._world.get_level_bbs(carla.CityObjectLabel.Vehicles)   # 获取场景中所有车辆的bounding boxes
		walker_bbox_list = self._world.get_level_bbs(carla.CityObjectLabel.Pedestrians) # 获取场景中所有行人的bounding boxes
		# print(f"[DEBUG] total vehicle_bbox_list = {len(vehicle_bbox_list)}")
		if self._scale_bbox:  # 执行 将在ev_loc周围的车辆和行人筛选出来
			vehicles = self._get_surrounding_actors(vehicle_bbox_list, is_within_distance, 1.0)  # 筛选出周围感兴趣的车辆,vehicles=[[位置+朝向,占位符[0,0,0],尺寸一半],[位置+朝向,占位符[0,0,0],尺寸一半]...]
			# print(f"[DEBUG] filtered vehicles = {len(vehicles)}")
			walkers = self._get_surrounding_actors(walker_bbox_list, is_within_distance, 2.0)    # 筛选出周围感兴趣的行人，2代表将行人的真实尺寸的长和宽放大了2倍，为了看起来更大
		else:
			vehicles = self._get_surrounding_actors(vehicle_bbox_list, is_within_distance)
			walkers = self._get_surrounding_actors(walker_bbox_list, is_within_distance)
		"""


		# ===================== 获取周围车辆/行人 =====================
		all_vehicle_actors = self._world.get_actors().filter('*vehicle*')
		all_walker_actors = self._world.get_actors().filter('*walker*')

		# print(f"[DEBUG] world actor vehicles = {len(all_vehicle_actors)}")
		# print(f"[DEBUG] world actor walkers = {len(all_walker_actors)}")

		vehicles = []
		walkers = []

		for actor in all_vehicle_actors:
			if actor.id == self._parent_actor.id:
				continue  # 排除自车

			actor_tf = actor.get_transform()
			actor_loc = actor_tf.location

			c_distance = abs(ev_loc.x - actor_loc.x) < self._distance_threshold and \
						abs(ev_loc.y - actor_loc.y) < self._distance_threshold and \
						abs(ev_loc.z - actor_loc.z) < 8.0

			if c_distance:
				bb = actor.bounding_box
				bb_loc = bb.location
				bb_ext = carla.Vector3D(bb.extent.x, bb.extent.y, bb.extent.z)

				if self._scale_bbox:
					bb_ext = bb_ext * 1.0
					bb_ext.x = max(bb_ext.x, 0.8)
					bb_ext.y = max(bb_ext.y, 0.8)

				vehicles.append((actor_tf, bb_loc, bb_ext))

		for actor in all_walker_actors:
			actor_tf = actor.get_transform()
			actor_loc = actor_tf.location

			c_distance = abs(ev_loc.x - actor_loc.x) < self._distance_threshold and \
						abs(ev_loc.y - actor_loc.y) < self._distance_threshold and \
						abs(ev_loc.z - actor_loc.z) < 8.0

			if c_distance:
				bb = actor.bounding_box
				bb_loc = bb.location
				bb_ext = carla.Vector3D(bb.extent.x, bb.extent.y, bb.extent.z)

				if self._scale_bbox:
					bb_ext = bb_ext * 2.0
					bb_ext.x = max(bb_ext.x, 0.8)
					bb_ext.y = max(bb_ext.y, 0.8)

				walkers.append((actor_tf, bb_loc, bb_ext))

		# print(f"[DEBUG] filtered vehicles = {len(vehicles)}")
		# print(f"[DEBUG] filtered walkers = {len(walkers)}")








		tl_green = TrafficLightHandler.get_stopline_vtx(ev_loc, 0)  # 绿灯 [carla.Location(x1, y1, z1), carla.Location(x2, y2, z2), ...] 点表示在 Carla 世界坐标系下的停车线（Stopline）顶点坐标
		tl_yellow = TrafficLightHandler.get_stopline_vtx(ev_loc, 1) # 黄灯 [carla.Location(x1, y1, z1), carla.Location(x2, y2, z2), ...] 点表示在 Carla 世界坐标系下的停车线（Stopline）顶点坐标
		tl_red = TrafficLightHandler.get_stopline_vtx(ev_loc, 2)    # 红灯 [carla.Location(x1, y1, z1), carla.Location(x2, y2, z2), ...] 点表示在 Carla 世界坐标系下的停车线（Stopline）顶点坐标
		stops = self._get_stops(self._criteria_stop)  # 1.如果没有满足条件的停车标志，返回空列表 [] 2.否则返回一个列表，包含停车点的详细信息(用于后续路径规划或停车行为判断)

		self._history_queue.append((vehicles, walkers, tl_green, tl_yellow, tl_red, stops))  # 将当前帧的可观察物体加入历史轨迹中 便于生成带时间线的语义图(动态BEV)

		M_warp = self._get_warp_transform(ev_loc, ev_rot)  # 仿射变换矩阵，这是核心，这个变换矩阵就使得自车中心在BEV中的位置、自车朝向图像向上

		# objects with history  周围车辆、周围行人、绿灯、黄灯、红灯、停止线的 BEVmask 均为列表 均是历史的加当前的
		vehicle_masks, walker_masks, tl_green_masks, tl_yellow_masks, tl_red_masks, stop_masks \
			= self._get_history_masks(M_warp)

		############################### mask ###############################
		# road_mask(道路), lane_mask道路线(实线 虚线)
		road_mask = cv.warpAffine(self._road, M_warp, (self._width, self._width)).astype(np.bool)  # 0 1 格式
		lane_mask_all = cv.warpAffine(self._lane_marking_all, M_warp, (self._width, self._width)).astype(np.bool)
		lane_mask_broken = cv.warpAffine(self._lane_marking_white_broken, M_warp,
										 (self._width, self._width)).astype(np.bool)

		# route_mask  渲染路径路线图层  route_plan是全局规划路径
		route_mask = np.zeros([self._width, self._width], dtype=np.uint8)
		route_in_pixel = np.array([[self._world_to_pixel(wp.transform.location)]   # 首先将在carla世界坐标系中的全局路径点坐标全部转换到图像中的像素坐标[x,y]
								   for wp, _ in route_plan[0:80]])  # 将route_lane(全局路径)中的前80个路径点渲染成像素路径
		route_warped = cv.transform(route_in_pixel, M_warp) # 然后再转换到BEV中
		cv.polylines(route_mask, [np.round(route_warped).astype(np.int32)], False, 1, thickness=16)  # 绘制为厚度为16的折线，形成路径mask
		route_mask = route_mask.astype(np.bool)  # 这是全局参考路径的mask

		# ev_mask 渲染自车mask 也是转换到bev中了
		ev_mask = self._get_mask_from_actor_list([(ev_transform, ev_bbox.location, ev_bbox.extent)], M_warp)
		ev_mask_col = self._get_mask_from_actor_list([(ev_transform, ev_bbox.location,
													   ev_bbox.extent*self._scale_mask_col)], M_warp)

		############################### image ###############################
		# 绘制可视化图像BEV image
		image = np.zeros([self._width, self._width, 3], dtype=np.uint8)

		image[road_mask] = COLOR_ALUMINIUM_5       # 道路 -> 深灰色偏蓝  在1的位置上色(只在 road_mask==True 的地方上色)
		# image[route_mask] = COLOR_ALUMINIUM_3      # 参考路径 -> 灰色
		# image[lane_mask_all] = COLOR_MAGENTA       # 实线 -> 品红色
		# image[lane_mask_broken] = COLOR_MAGENTA_2  # 虚线 -> 浅品红色

		# 构建最终语义mask
		# 绘制停止线和红绿灯
		# h_len = len(self._history_idx)-1 # 代表的是历史信息是3帧 那么再加上当前帧 一共是4帧
		# for i, mask in enumerate(stop_masks):
		# 	image[mask] = tint(COLOR_YELLOW_2, (h_len-i)*0.2)
		# for i, mask in enumerate(tl_green_masks):
		# 	image[mask] = tint(COLOR_GREEN, (h_len-i)*0.2)
		# for i, mask in enumerate(tl_yellow_masks):
		# 	image[mask] = tint(COLOR_YELLOW, (h_len-i)*0.2)
		# for i, mask in enumerate(tl_red_masks):
		# 	image[mask] = tint(COLOR_RED, (h_len-i)*0.2)

		# 绘制4帧(当前帧+过去3帧)
		# for i, mask in enumerate(vehicle_masks):   # 周围车是蓝色
		# 	image[mask] = tint(COLOR_BLUE, (h_len-i)*0.2)
		# for i, mask in enumerate(walker_masks):    # 周围行人
		# 	image[mask] = tint(COLOR_CYAN, (h_len-i)*0.2)
		# 只绘制最后一帧（当前帧）
		image[vehicle_masks[-1]] = COLOR_BLUE
		image[walker_masks[-1]] = COLOR_CYAN

		image[ev_mask] = COLOR_WHITE  # 自车是白色
		# image[ev_mask] = COLOR_YELLOW   # 自车是黄色

		# masks
		c_road = road_mask * 255
		c_route = route_mask * 255
		c_lane = lane_mask_all * 255
		c_lane[lane_mask_broken] = 120

		# masks with history
		c_tl_history = []
		for i in range(len(self._history_idx)):  # 过去的第1、6、11、16帧
			c_tl = np.zeros([self._width, self._width], dtype=np.uint8)
			c_tl[tl_green_masks[i]] = 80
			c_tl[tl_yellow_masks[i]] = 170
			c_tl[tl_red_masks[i]] = 255
			c_tl[stop_masks[i]] = 255
			c_tl_history.append(c_tl)

		c_vehicle_history = [m*255 for m in vehicle_masks]
		c_walker_history = [m*255 for m in walker_masks]

		masks = np.stack((c_road, c_route, c_lane, *c_vehicle_history, *c_walker_history, *c_tl_history), axis=2)
		masks = np.transpose(masks, [2, 0, 1])

		############################################ 自开发功能 ############################################

		# ================== 绘制周围车辆的中心点和航向角 ==================
		vehicles_info = self.draw_actor_centers_bev(image, vehicles, M_warp, ev_loc, ev_rot_yaw_rad, color=(0, 255, 255))
		walkers_info = self.draw_actor_centers_bev(image, walkers, M_warp, ev_loc, ev_rot_yaw_rad, color=(255, 255, 0))

		# ================= 显示前20个路径点 —— 圆点形式 ==================
		# for i in range(min(20, len(route_warped))):
		# 	pt = route_warped[i][0]  # shape 是 [N, 1, 2]
		# 	cv.circle(image,
		# 			  center=(int(round(pt[0])), int(round(pt[1]))),
		# 			  radius=1,
		# 			  color=(255, 0, 0),  # 红色圆点，RGB
		# 			  thickness=2)  # 实心圆

		# ================= 显示前20个路径点 —— 线段形式 ==================
		pts = np.array([pt[0] for pt in route_warped[:min(20, len(route_warped))]], dtype=np.int32)
		# 绘制折线
		# cv.polylines(
		# 	image,
		# 	[pts],
		# 	isClosed=False,  # 不闭合
		# 	color=(255, 0, 0),  # 红色
		# 	thickness=2  # 线宽
		# )

		# ================= 绘制路径点方向箭头 =================
		route_pixels = []  # 用来存储像素坐标
		route_relative_yaw_rad = []  # 用来存储像素坐标
		for i in range(min(20, len(route_warped))):
			pt = route_warped[i][0]  # [x, y] 像素坐标
			route_pixels.append(pt.tolist())  # 保存路径的像素点坐标
			yaw = route_plan[i][0].transform.rotation.yaw  # 获取路径点原始朝向角（度）
			yaw_rad = math.radians(yaw)  # 转换为弧度

			# ⬅️ 转为自车坐标系下的相对角度
			relative_yaw_rad = yaw_rad - ev_rot_yaw_rad
			route_relative_yaw_rad.append(relative_yaw_rad)  # 保存路径的相对自车的角度  负值代表路径点偏左  正值代表路径点偏右 均在BEV

			# ⬆️ 在 BEV 图像中，自车朝向为向上（负 y），所以箭头要用 sin/cos 做旋转
			arrow_len = 10
			end_x = int(round(pt[0] + arrow_len * math.sin(relative_yaw_rad)))
			end_y = int(round(pt[1] - arrow_len * math.cos(relative_yaw_rad)))

			# 显示路径点方向箭头
			# cv.arrowedLine(image,
			# 			   (int(round(pt[0])), int(round(pt[1]))),
			# 			   (end_x, end_y),
			# 			   color=(0, 255, 0),  # 绿色箭头
			# 			   thickness=1,
			# 			   tipLength=0.3)
		# 该20个路径点的世界坐标
		route_world = np.array([[wp.transform.location.x, wp.transform.location.y]
								for wp, _ in route_plan[0:20]], dtype=np.float32).tolist()

		obs_dict = {'rendered': image,
					'masks': masks,
					'vehicles_info': vehicles_info,
					'walkers_info': walkers_info,
					'route_pixels': route_pixels,'route_world': route_world,
					'route_relative_yaw_rad': route_relative_yaw_rad}  # image就是bev mask是bev_image的语义形式

		return obs_dict

	def _get_history_masks(self, M_warp):
		qsize = len(self._history_queue)
		vehicle_masks, walker_masks, tl_green_masks, tl_yellow_masks, tl_red_masks, stop_masks = [], [], [], [], [], []
		for idx in self._history_idx:
			idx = max(idx, -1 * qsize)

			vehicles, walkers, tl_green, tl_yellow, tl_red, stops = self._history_queue[idx]  # 取出来

			vehicle_masks.append(self._get_mask_from_actor_list(vehicles, M_warp))     # 周围感兴趣车辆的mask 192x192 也就是BEV大小
			walker_masks.append(self._get_mask_from_actor_list(walkers, M_warp))       # 周围感兴趣行人的mask 192x192 也就是BEV大小
			tl_green_masks.append(self._get_mask_from_stopline_vtx(tl_green, M_warp))  # 绿灯的mask          192x192 也就是BEV大小
			tl_yellow_masks.append(self._get_mask_from_stopline_vtx(tl_yellow, M_warp))# 黄灯的mask          192x192 也就是BEV大小
			tl_red_masks.append(self._get_mask_from_stopline_vtx(tl_red, M_warp))      # 红灯的mask          192x192 也就是BEV大小
			stop_masks.append(self._get_mask_from_actor_list(stops, M_warp))           # 停止线的mask        192x192 也就是BEV大小

		return vehicle_masks, walker_masks, tl_green_masks, tl_yellow_masks, tl_red_masks, stop_masks

	def _get_mask_from_stopline_vtx(self, stopline_vtx, M_warp):
		mask = np.zeros([self._width, self._width], dtype=np.uint8)
		for sp_locs in stopline_vtx:
			stopline_in_pixel = np.array([[self._world_to_pixel(x)] for x in sp_locs])
			stopline_warped = cv.transform(stopline_in_pixel, M_warp)
			stopline_warped = np.round(stopline_warped).astype(np.int32)
			cv.line(mask, tuple(stopline_warped[0, 0]), tuple(stopline_warped[1, 0]),
					color=1, thickness=6)
		return mask.astype(np.bool)

	def _get_mask_from_actor_list(self, actor_list, M_warp):
		mask = np.zeros([self._width, self._width], dtype=np.uint8)  # 生成一个大小为192x192 BEV图尺寸的全 0 图像（黑色图），准备在上面画物体的位置。
		for actor_transform, bb_loc, bb_ext in actor_list:   # bb_ext为b_box的真实尺寸/2 -> l/2 w/2 h/2

			corners = [carla.Location(x=-bb_ext.x, y=-bb_ext.y),
					   carla.Location(x=bb_ext.x, y=-bb_ext.y),
					   carla.Location(x=bb_ext.x, y=0),
					   carla.Location(x=bb_ext.x, y=bb_ext.y),
					   carla.Location(x=-bb_ext.x, y=bb_ext.y)]
			corners = [bb_loc + corner for corner in corners]  # bb_loc = [0,0,0]

			# 世界坐标系下的角点
			corners = [actor_transform.transform(corner) for corner in corners]
			corners_in_pixel = np.array([[self._world_to_pixel(corner)] for corner in corners])  # 把这些世界坐标系下的角点转换成图像中对应的像素位置
			corners_warped = cv.transform(corners_in_pixel, M_warp)  # 通过仿射变换转换到BEV坐标系

			# 在空白的 mask 上，把这个物体的区域（凸多边形）用值为 1 的像素填上。
			cv.fillConvexPoly(mask, np.round(corners_warped).astype(np.int32), 1)
		# 最终拿到的是一个布尔类型的 BEV 视角下的“遮罩图”：True → 表示这个区域被物体（车辆、人）占据;False → 表示空白区域.
		# 最终返回的是一个 BEV 图像大小的 mask
		return mask.astype(np.bool)

	@staticmethod
	def _get_surrounding_actors(bbox_list, criterium, scale=None):
		actors = []   # 创建空列表用于存储周围感兴趣区域内的物体
		for bbox in bbox_list:  # 逐步检查每一个车辆或者行人是否满足要求
			is_within_distance = criterium(bbox)  # 筛选规则->在自车附近且不是自车
			if is_within_distance:  # 若满足要求，即是周围车且不是自车
				bb_loc = carla.Location()  # 空的 carla.Location() 对象，即其 x, y, z 默认为 0m。当前代码无实际意义，作为占位符使用
				bb_ext = carla.Vector3D(bbox.extent)  # bbox 的尺寸（extent 是半长、半宽、半高），单位m
				if scale is not None:  # 如果进行缩放
					bb_ext = bb_ext * scale  # 按照尺寸进行缩放
					bb_ext.x = max(bb_ext.x, 0.8)  # 设置最小尺寸为0.8 长度的一半
					bb_ext.y = max(bb_ext.y, 0.8)  # 设置最小尺寸为0.8 宽度的一半
				"""
				bb_ext.x  # 车辆沿前后方向的一半长度（前轴到中心）
				bb_ext.y  # 车辆沿左右方向的一半宽度（中心到侧边）
				bb_ext.z  # 车辆沿上下方向的一半高度（地面到中心）
				"""

				# 每个 actor 被记录为一个三元组：[Transform, Location, Extent],[[[x,y,z],[roll,pitch,yaw]],[0,0,0],[l/2,w/2,h/2]]
				# bbox.location单位m, bbox.rotation单位度,均为carla世界坐标系
				actors.append((carla.Transform(bbox.location, bbox.rotation), bb_loc, bb_ext))
		return actors

	def _get_warp_transform(self, ev_loc, ev_rot):
		ev_loc_in_px = self._world_to_pixel(ev_loc)  # 将自车在carla世界坐标系下的位置转换到了图像中,单位:像素,[x,y]
		yaw = np.deg2rad(ev_rot.yaw)                 # 单位:弧度 carla中,ev_rot.yaw=0 朝东(x+) ev_rot.yaw=90 朝南(y+) ev_rot.yaw=-90 朝北(y-) z轴垂直地面向上

		# 定义自车的坐标系:前向和右向坐标轴
		forward_vec = np.array([np.cos(yaw), np.sin(yaw)])  # 实际上这里可以认为是单位向量  [cos,sin] 假设现在自车的yaw为90度,那么forward_vec=[0,1],这就代表自车的前行方向为南向
		right_vec = np.array([np.cos(yaw + 0.5*np.pi), np.sin(yaw + 0.5*np.pi)])  # [-sin,cos] 此时right_vec=[-1,0],这就代表自车的右边为西向

		# BEV图像的三个顶点 这三个顶点会根据自车的图像坐标系下的位置的变化而变化,但永远是自车周围的固定区域
		bottom_left = ev_loc_in_px - self._pixels_ev_to_bottom * forward_vec - (0.5*self._width) * right_vec
		top_left = ev_loc_in_px + (self._width-self._pixels_ev_to_bottom) * forward_vec - (0.5*self._width) * right_vec
		top_right = ev_loc_in_px + (self._width-self._pixels_ev_to_bottom) * forward_vec + (0.5*self._width) * right_vec

		# 将三个二维坐标点（每个是 [x, y]）组合成一个 3×2 的矩阵，用于图像的仿射变换（Affine Transform）。
		src_pts = np.stack((bottom_left, top_left, top_right), axis=0).astype(np.float32)
		# 定义BEV图像的显示形式 正是因为dst_pts的存在才确定了BEV图像的表现形式
		dst_pts = np.array([[0, self._width-1],
							[0, 0],
							[self._width-1, 0]], dtype=np.float32)
		return cv.getAffineTransform(src_pts, dst_pts)

	def _world_to_pixel(self, location, projective=False):
		"""Converts the world coordinates to pixel coordinates"""
		"""这里的图像坐标系实际上就是高精地图"""
		x = self._pixels_per_meter * (location.x - self._world_offset[0])  # 单位：像素
		y = self._pixels_per_meter * (location.y - self._world_offset[1])  # 单位：像素

		if projective:
			p = np.array([x, y, 1], dtype=np.float32)
		else:  # 执行
			p = np.array([x, y], dtype=np.float32)
		return p

	def _world_to_pixel_width(self, width):
		"""Converts the world units to pixel units"""
		return self._pixels_per_meter * width

	def clean(self):
		self._parent_actor = None
		self._world = None
		self._history_queue.clear()
