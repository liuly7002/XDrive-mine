import os
import json
import datetime
import pathlib
import time
import cv2
import carla
from collections import deque
import math
from collections import OrderedDict

import torch
import carla
import numpy as np
from PIL import Image
from torchvision import transforms as T

from leaderboard.autoagents import autonomous_agent

from TCP.model import TCP
from TCP.config import GlobalConfig
from team_code.planner import RoutePlanner


# liulei
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from leaderboard.utils.route_manipulation import downsample_route
from roach.obs_manager.birdview.chauffeurnet import ObsManager
from omegaconf import OmegaConf
from roach.criteria import run_stop_sign



SAVE_PATH = os.environ.get('SAVE_PATH', None)


def get_entry_point():
	return 'TCPAgent'


class TCPAgent(autonomous_agent.AutonomousAgent):
	def setup(self, path_to_conf_file):
		self.track = autonomous_agent.Track.SENSORS
		self.alpha = 0.3
		self.status = 0
		self.steer_step = 0
		self.last_moving_status = 0
		self.last_moving_step = -1
		self.last_steers = deque()

		self.config_path = path_to_conf_file
		self.step = -1
		self.wall_start = time.time()
		self.initialized = False

		self.config = GlobalConfig()
		self.net = TCP(self.config)


		# ckpt = torch.load(path_to_conf_file)
		ckpt = torch.load(path_to_conf_file, map_location='cuda:0')
		ckpt = ckpt["state_dict"]
		new_state_dict = OrderedDict()
		for key, value in ckpt.items():
			new_key = key.replace("model.","")
			new_state_dict[new_key] = value
		self.net.load_state_dict(new_state_dict, strict = False)
		self.net.cuda()
		self.net.eval()

		self.takeover = False
		self.stop_time = 0
		self.takeover_time = 0

		self.save_path = None
		self._im_transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

		self.last_steers = deque()
		if SAVE_PATH is not None:
			now = datetime.datetime.now()
			string = pathlib.Path(os.environ['ROUTES']).stem + '_'
			string += '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))

			print (string)

			self.save_path = pathlib.Path(os.environ['SAVE_PATH']) / string
			self.save_path.mkdir(parents=True, exist_ok=False)

			# (self.save_path / 'rgb').mkdir()
			(self.save_path / 'meta').mkdir()
			(self.save_path / 'bev').mkdir()
			# (self.save_path / 'bev_waypoints').mkdir()
			(self.save_path / 'bev_segmentation').mkdir()

		# ==================================== 新增 ====================================
		cfg = OmegaConf.load("roach/config/config_agent.yaml")
		cfg = OmegaConf.to_container(cfg)
		self.cfg = cfg
		# ==================================== 新增 ====================================


	def _init(self):
		self._route_planner = RoutePlanner(4.0, 50.0)
		self._route_planner.set_route(self._global_plan, True)

		self._command_planner = RoutePlanner(7.5, 25.0, 257)
		self._command_planner.set_route(self._global_plan, True)

		self.initialized = True

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.net.to(self.device)

		# ==================================== 新增 ====================================
		self._world = CarlaDataProvider.get_world()
		self._ego_vehicle = CarlaDataProvider.get_ego()
		self._criteria_stop = run_stop_sign.RunStopSign(self._world)
		self.birdview_obs_manager = ObsManager(self.cfg['obs_configs']['birdview'], self._criteria_stop)
		self.birdview_obs_manager.attach_ego_vehicle(self._ego_vehicle)
		# ==================================== 新增 ====================================

		# self.emergency_brake = False

	# ==================================== 新增 ====================================
	def _truncate_global_route_till_local_target(self, windows_size=5):
		ev_location = self._ego_vehicle.get_location()   # 从carla模拟器中获取自车当前的位置(x,y,z)
		closest_idx = 0  # 初始化"距离自车最近的全局路径点索引"为0.
		for i in range(len(self._global_route)-1):
			if i > windows_size:  # ；遍历全局路径中的0,1,2,3,4,5的六个点
				break

			loc0 = self._global_route[i][0].transform.location
			loc1 = self._global_route[i+1][0].transform.location

			wp_dir = loc1 - loc0         # 从当前路径点 i 指向下一个路径点 i+1 的方向向量.
			wp_veh = ev_location - loc0  # 从路径点 i 到当前车辆位置的向量.
			dot_ve_wp = wp_veh.x * wp_dir.x + wp_veh.y * wp_dir.y + wp_veh.z * wp_dir.z
			# 对两个向量取点积，用来判断它们的相对方向关系：
			# dot_ve_wp > 0：说明车辆位置相对于路径段起点，在路径方向的“前方”;
			# dot_ve_wp < 0：说明车辆还在该段的"后方".

			if dot_ve_wp > 0:
				closest_idx = i+1
		if closest_idx > 0:
			self._last_route_location = carla.Location(self._global_route[0][0].transform.location)

		self._global_route = self._global_route[closest_idx:]  # 将 _global_route 从 closest_idx 开始截断,即“删除”掉车辆已经通过的路径点,只保留从当前位置往前的路径.

	def set_global_plan(self, global_plan_gps, global_plan_world_coord, wp_route):
		"""
		Set the plan (route) for the agent
		"""
		self._global_route = wp_route
		ds_ids = downsample_route(global_plan_world_coord, 50)
		self._global_plan = [global_plan_gps[x] for x in ds_ids]
		self._global_plan_world_coord = [(global_plan_world_coord[x][0], global_plan_world_coord[x][1]) for x in ds_ids]

		self._plan_gps_HACK = global_plan_gps
		self._plan_HACK = global_plan_world_coord
	# ==================================== 新增 ====================================

	def _get_position(self, tick_data):
		gps = tick_data['gps']
		gps = (gps - self._route_planner.mean) * self._route_planner.scale
		# gps = (gps - self._command_planner.mean) * self._command_planner.scale

		return gps

	def sensors(self):
				return [
				{
					'type': 'sensor.camera.rgb',
					'x': -1.5, 'y': 0.0, 'z':2.0,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'width': 900, 'height': 256, 'fov': 100,
					'id': 'rgb'
					},
				{
					'type': 'sensor.camera.rgb',
					'x': 0.0, 'y': 0.0, 'z': 40.0,
					'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
					# 'width': 512, 'height': 512, 'fov': 5 * 10.0,
					'width': 1024, 'height': 1024, 'fov': 5 * 10.0,
					'id': 'bev'
					},	
				{
					'type': 'sensor.other.imu',
					'x': 0.0, 'y': 0.0, 'z': 0.0,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'sensor_tick': 0.05,
					'id': 'imu'
					},
				{
					'type': 'sensor.other.gnss',
					'x': 0.0, 'y': 0.0, 'z': 0.0,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'sensor_tick': 0.01,
					'id': 'gps'
					},
				{
					'type': 'sensor.speedometer',
					'reading_frequency': 20,
					'id': 'speed'
					}
				]

	def tick(self, input_data):
		self.step += 1

		# ==================================== 新增 ====================================
		self._truncate_global_route_till_local_target()
		birdview_obs = self.birdview_obs_manager.get_observation(self._global_route)  # BEV
		# ==================================== 新增 ====================================

		rgb = cv2.cvtColor(input_data['rgb'][1][:, :, :3], cv2.COLOR_BGR2RGB)
		bev = cv2.cvtColor(input_data['bev'][1][:, :, :3], cv2.COLOR_BGR2RGB)
		gps = input_data['gps'][1][:2]
		speed = input_data['speed'][1]['speed']   # 车速
		compass = input_data['imu'][1][-1]        # 航向角(yaw)

		if (math.isnan(compass) == True): #It can happen that the compass sends nan for a few frames
			compass = 0.0

		result = {
				'rgb': rgb,
				'gps': gps,
				'speed': speed,
				'compass': compass,
				'bev': bev
				}
		
		pos = self._get_position(result)
		result['gps'] = pos
		next_wp, next_cmd = self._route_planner.run_step(pos)
		result['next_command'] = next_cmd.value


		theta = compass + np.pi/2
		R = np.array([
			[np.cos(theta), -np.sin(theta)],
			[np.sin(theta), np.cos(theta)]
			])

		local_command_point = np.array([next_wp[0]-pos[0], next_wp[1]-pos[1]])
		local_command_point = R.T.dot(local_command_point)
		result['target_point'] = tuple(local_command_point)

		return result, birdview_obs['rendered'], birdview_obs

	@torch.no_grad()
	def run_step(self, input_data, timestamp):

		if not self.initialized:
			self._init()

		tick_data, rendered, veh_wal_info = self.tick(input_data)

		if self.step < self.config.seq_len:
			rgb = self._im_transform(tick_data['rgb']).unsqueeze(0)
			control = carla.VehicleControl()
			control.steer = 0.0
			control.throttle = 0.0
			control.brake = 0.0
			return control

		gt_velocity = torch.FloatTensor([tick_data['speed']]).to('cuda', dtype=torch.float32)
		command = tick_data['next_command']
		if command < 0:
			command = 4
		command -= 1
		assert command in [0, 1, 2, 3, 4, 5]
		cmd_one_hot = [0] * 6
		cmd_one_hot[command] = 1
		cmd_one_hot = torch.tensor(cmd_one_hot).view(1, 6).to('cuda', dtype=torch.float32)
		speed = torch.FloatTensor([float(tick_data['speed'])]).view(1,1).to('cuda', dtype=torch.float32)
		speed = speed / 12
		rgb = self._im_transform(tick_data['rgb']).unsqueeze(0).to('cuda', dtype=torch.float32)

		tick_data['target_point'] = [torch.FloatTensor([tick_data['target_point'][0]]), torch.FloatTensor([tick_data['target_point'][1]])]
		target_point = torch.stack(tick_data['target_point'], dim=1).to('cuda', dtype=torch.float32)

		state = torch.cat([speed, target_point, cmd_one_hot], 1)

		# ========== 将参考路径的点(<=20个)从新坐标系下转换到自车坐标系,然后将不足20个点的路径padding到20个点 ==========
		ego_theta = tick_data['compass']
		ego_x,ego_y = veh_wal_info['vehicles_info'][0]['ev_loc_info_world'][:2]
		routepoints = []
		R = np.array([
			[np.cos(np.pi / 2 + ego_theta), -np.sin(np.pi / 2 + ego_theta)],
			[np.sin(np.pi / 2 + ego_theta), np.cos(np.pi / 2 + ego_theta)]
		])
		for point in veh_wal_info['route_world']:
			px, py = point
			local_route_point = np.array([py - ego_y, px - ego_x])
			local_route_point = R.T.dot(local_route_point)
			routepoints.append(local_route_point)
		routepoints = np.array(routepoints)  # [N, 2]
		actual_len = routepoints.shape[0]
		if actual_len == 0:
			# 避免空数组报错，用零填充
			routepoints = np.zeros((20, 2))
			actual_len = 0
		elif actual_len < 20:
			pad_len = 20 - actual_len
			pad = np.tile(routepoints[-1:], (pad_len, 1))
			routepoints = np.vstack([routepoints, pad])  # pad最后一个路径点
		routepoints = torch.tensor(routepoints, dtype=torch.float32).unsqueeze(0).to(self.device)
		actual_len = torch.tensor([actual_len], dtype=torch.long).to(self.device)
		# ========== 将参考路径的点(<=20个)从新坐标系下转换到自车坐标系,然后将不足20个点的路径padding到20个点 ==========

		pred= self.net(rgb, state, target_point, routepoints, actual_len)

		########################################## 实验结果展示_01: 语义分割BEV ##########################################
		pred_wp_ego = pred['pred_wp'][0].detach().cpu().numpy()  # (T,2)  pred_wp.shape = (T,2)，列0: 前向，列1: 左向
		# 自车像素坐标
		ego_px, ego_py = 96, 152
		# 像素/米
		pixels_per_meter = 5.0
		# 映射到图像坐标系
		wp_px = ego_px + pred_wp_ego[:, 0] * pixels_per_meter
		wp_py = ego_py + pred_wp_ego[:, 1] * pixels_per_meter
		wp_pixels = np.stack([wp_px, wp_py], axis=1).astype(np.int32)
		# 绘制轨迹
		bev_draw = rendered.copy()
		cv2.polylines(bev_draw, [wp_pixels], isClosed=False, color=(0, 255, 0), thickness=2)
		rendered = bev_draw
		########################################## 实验结果展示_01: 语义分割BEV ##########################################

		steer_ctrl, throttle_ctrl, brake_ctrl, metadata = self.net.process_action(pred, tick_data['next_command'], gt_velocity, target_point)

		steer_traj, throttle_traj, brake_traj, metadata_traj = self.net.control_pid(pred['pred_wp'], gt_velocity, target_point)
		if brake_traj < 0.05: brake_traj = 0.0
		if throttle_traj > brake_traj: brake_traj = 0.0

		self.pid_metadata = metadata_traj
		control = carla.VehicleControl()

		if self.status == 0:
			self.alpha = 0.3
			self.pid_metadata['agent'] = 'traj'
			control.steer = np.clip(self.alpha*steer_ctrl + (1-self.alpha)*steer_traj, -1, 1)
			control.throttle = np.clip(self.alpha*throttle_ctrl + (1-self.alpha)*throttle_traj, 0, 0.75)
			control.brake = np.clip(self.alpha*brake_ctrl + (1-self.alpha)*brake_traj, 0, 1)
		else:
			self.alpha = 0.3
			self.pid_metadata['agent'] = 'ctrl'
			control.steer = np.clip(self.alpha*steer_traj + (1-self.alpha)*steer_ctrl, -1, 1)
			control.throttle = np.clip(self.alpha*throttle_traj + (1-self.alpha)*throttle_ctrl, 0, 0.75)
			control.brake = np.clip(self.alpha*brake_traj + (1-self.alpha)*brake_ctrl, 0, 1)


		self.pid_metadata['steer_ctrl'] = float(steer_ctrl)
		self.pid_metadata['steer_traj'] = float(steer_traj)
		self.pid_metadata['throttle_ctrl'] = float(throttle_ctrl)
		self.pid_metadata['throttle_traj'] = float(throttle_traj)
		self.pid_metadata['brake_ctrl'] = float(brake_ctrl)
		self.pid_metadata['brake_traj'] = float(brake_traj)

		if control.brake > 0.5:
			control.throttle = float(0)

		if len(self.last_steers) >= 20:
			self.last_steers.popleft()
		self.last_steers.append(abs(float(control.steer)))
		#chech whether ego is turning
		# num of steers larger than 0.1
		num = 0
		for s in self.last_steers:
			if s > 0.10:
				num += 1
		if num > 10:
			self.status = 1
			self.steer_step += 1

		else:
			self.status = 0

		self.pid_metadata['status'] = self.status

		if SAVE_PATH is not None and self.step % 10 == 0:
			self.save(tick_data, rendered)

		return control

	def save(self, tick_data, render_img):
		frame = self.step // 10

		# 实验结果展示_01: 语义分割BEV
		# Image.fromarray(render_img).save(self.save_path / 'bev_segmentation' / ('%04d.png' % frame))
		# Image.fromarray(tick_data['bev']).save(self.save_path / 'bev' / ('%04d.png' % frame))

		# Image.fromarray(tick_data['rgb']).save(self.save_path / 'rgb' / ('%04d.png' % frame))

		outfile = open(self.save_path / 'meta' / ('%04d.json' % frame), 'w')
		json.dump(self.pid_metadata, outfile, indent=4)
		outfile.close()

	def destroy(self):
		del self.net
		torch.cuda.empty_cache()