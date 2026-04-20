import os
import json
import datetime
import pathlib
import time
import cv2

import torch
import carla
import numpy as np
from PIL import Image

from leaderboard.autoagents import autonomous_agent
import numpy as np
from omegaconf import OmegaConf

from roach.criteria import run_stop_sign
from roach.obs_manager.birdview.chauffeurnet import ObsManager
from roach.utils.config_utils import load_entry_point
import roach.utils.transforms as trans_utils
from roach.utils.traffic_light import TrafficLightHandler

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from leaderboard.utils.route_manipulation import downsample_route
from agents.navigation.local_planner import RoadOption

from team_code.planner import RoutePlanner


SAVE_PATH = os.environ.get('SAVE_PATH', None)

def get_entry_point():
	return 'ROACHAgent'

def _numpy(carla_vector, normalize=False):
	result = np.float32([carla_vector.x, carla_vector.y])

	if normalize:
		return result / (np.linalg.norm(result) + 1e-4)

	return result


def _location(x, y, z):
	return carla.Location(x=float(x), y=float(y), z=float(z))


def get_xyz(_):
	return [_.x, _.y, _.z]


def _orientation(yaw):
	return np.float32([np.cos(np.radians(yaw)), np.sin(np.radians(yaw))])


def get_collision(p1, v1, p2, v2):
	A = np.stack([v1, -v2], 1)
	b = p2 - p1

	if abs(np.linalg.det(A)) < 1e-3:
		return False, None

	x = np.linalg.solve(A, b)
	collides = all(x >= 0) and all(x <= 1) # how many seconds until collision

	return collides, p1 + x[0] * v1


class ROACHAgent(autonomous_agent.AutonomousAgent):
	def setup(self, path_to_conf_file, ckpt="roach/log/ckpt_11833344.pth"):
		self._render_dict = None
		self.supervision_dict = None
		self._ckpt = ckpt
		cfg = OmegaConf.load(path_to_conf_file)
		cfg = OmegaConf.to_container(cfg)
		self.cfg = cfg
		self._obs_configs = cfg['obs_configs']
		self._train_cfg = cfg['training']
		self._policy_class = load_entry_point(cfg['policy']['entry_point'])
		self._policy_kwargs = cfg['policy']['kwargs']
		if self._ckpt is None:
			self._policy = None
		else:
			self._policy, self._train_cfg['kwargs'] = self._policy_class.load(self._ckpt)
			self._policy = self._policy.eval()
		self._wrapper_class = load_entry_point(cfg['env_wrapper']['entry_point'])
		self._wrapper_kwargs = cfg['env_wrapper']['kwargs']

		self.track = autonomous_agent.Track.SENSORS
		self.config_path = path_to_conf_file
		self.step = -1
		self.wall_start = time.time()
		self.initialized = False

		self._3d_bb_distance = 50

		self.prev_lidar = None

		self.save_path = None
		if SAVE_PATH is not None:
			now = datetime.datetime.now()
			string = pathlib.Path(os.environ['ROUTES']).stem + '_'
			string += '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))


			self.save_path = pathlib.Path(os.environ['SAVE_PATH']) / string
			self.save_path.mkdir(parents=True, exist_ok=False)

			(self.save_path / 'rgb').mkdir()
			(self.save_path / 'measurements').mkdir()
			(self.save_path / 'supervision').mkdir()
			(self.save_path / 'bev').mkdir()
			(self.save_path / 'bev_rgb').mkdir()


	def _init(self):
		self._waypoint_planner = RoutePlanner(4.0, 50)
		self._waypoint_planner.set_route(self._plan_gps_HACK, True)

		self._command_planner = RoutePlanner(7.5, 25.0, 257)
		self._command_planner.set_route(self._global_plan, True)

		self._route_planner = RoutePlanner(4.0, 50.0)
		self._route_planner.set_route(self._global_plan, True)

		self._world = CarlaDataProvider.get_world()
		self._map = self._world.get_map()
		self._ego_vehicle = CarlaDataProvider.get_ego()
		self._last_route_location = self._ego_vehicle.get_location()
		self._criteria_stop = run_stop_sign.RunStopSign(self._world)
		self.birdview_obs_manager = ObsManager(self.cfg['obs_configs']['birdview'], self._criteria_stop)
		self.birdview_obs_manager.attach_ego_vehicle(self._ego_vehicle)

		self.navigation_idx = -1


		# for stop signs
		self._target_stop_sign = None # the stop sign affecting the ego vehicle
		self._stop_completed = False # if the ego vehicle has completed the stop sign
		self._affected_by_stop = False # if the ego vehicle is influenced by a stop sign

		TrafficLightHandler.reset(self._world)
		print("initialized")

		self.initialized = True

	def _get_angle_to(self, pos, theta, target):
		R = np.array([
			[np.cos(theta), -np.sin(theta)],
			[np.sin(theta),  np.cos(theta)],
			])

		aim = R.T.dot(target - pos)
		angle = -np.degrees(np.arctan2(-aim[1], aim[0]))
		angle = 0.0 if np.isnan(angle) else angle 

		return angle
	

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

	def _get_position(self, tick_data):
		gps = tick_data['gps']
		gps = (gps - self._command_planner.mean) * self._command_planner.scale

		return gps

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

	def sensors(self):
		return [
				{
					'type': 'sensor.camera.rgb',
					'x': 6, 'y': 0.0, 'z':6.9,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'width': 900, 'height': 256, 'fov': 100,
					'id': 'rgb'
					},
				{
					'type': 'sensor.camera.rgb',
					'x': 0.0, 'y': 0.0, 'z': 70.0,
					'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
					'width': 768, 'height': 768, 'fov': 5 * 10.0,
					'id': 'bev_rgb'
					},
				{
					'type': 'sensor.other.imu',
					'x': 0.0, 'y': 0.0, 'z': 0.0,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'sensor_tick': 0.05,   # 表示每隔0.05s输出一次数据(20HZ)
					'id': 'imu'
					},
				{
					'type': 'sensor.other.gnss',
					'x': 0.0, 'y': 0.0, 'z': 0.0,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'sensor_tick': 0.01,  # 表示每隔0.01s输出一次数据(100HZ)
					'id': 'gps'
					},
				{
					'type': 'sensor.speedometer',   # 车速传感器
					'reading_frequency': 20,        # 每秒读取20次
					'id': 'speed'
					}
				]

	def tick(self, input_data, timestamp):
		self._truncate_global_route_till_local_target()

		birdview_obs = self.birdview_obs_manager.get_observation(self._global_route)  # BEV

		# 通过API获取自车的控制指令
		control = self._ego_vehicle.get_control()                          # 获取自车的控制命令
		throttle = np.array([control.throttle], dtype=np.float32)  # 油门值，通常范围是0~1的浮点数
		steer = np.array([control.steer], dtype=np.float32)        # 方向盘转角，通常范围是-1（左满）到1（右满）
		brake = np.array([control.brake], dtype=np.float32)        # 刹车力度，通常范围0~1
		gear = np.array([control.gear], dtype=np.float32)          # 档位，整数，但这里转成float32类型数组了

		# 通过API获取自车的位姿+车速
		ev_transform = self._ego_vehicle.get_transform()    # 自车的位置和姿态
		vel_w = self._ego_vehicle.get_velocity()            # 获取自车当前的速度向量，单位是 carla世界坐标系下的速度，vel_w是一个三维向量（x,y,z），表示车辆在世界坐标系中各方向的速度分量。
		vel_ev = trans_utils.vec_global_to_ref(vel_w, ev_transform.rotation)  # 自车坐标系： x->向前 y->向左 z->向上
		vel_xy = np.array([vel_ev.x, vel_ev.y], dtype=np.float32)
		# vel_ev.x：车辆在自己前进方向上的速度分量（m/s）
		# vel_ev.y：车辆在自己左侧方向上的速度分量（m/s）
		# vel_ev.z：车辆上下方向速度分量（m/s），通常很小或0，地面车辆一般忽略垂直速度


		self._criteria_stop.tick(self._ego_vehicle, timestamp)

		state_list = []
		state_list.append(throttle)  # 油门
		state_list.append(steer)     # 方向盘转角
		state_list.append(brake)     # 刹车
		state_list.append(gear)      # 档位
		state_list.append(vel_xy)    # 自车速度,x代表前行方向,y代表左侧方向
		state = np.concatenate(state_list)
		obs_dict = {
			'state': state.astype(np.float32),
			'birdview': birdview_obs['masks'],
		}

		# 通过自车安装的传感器来获取信息,也就是sensors函数定义的那些传感器
		rgb = cv2.cvtColor(input_data['rgb'][1][:, :, :3], cv2.COLOR_BGR2RGB)
		gps = input_data['gps'][1][:2]          # 自车在世界坐标系下的位置[纬度,经度]
		speed = input_data['speed'][1]['speed'] # 自车的速度,m/s,总线速度
		compass = input_data['imu'][1][-1]      # 自车航向角(compass heading)

		# target_gps: 下一个导航点的 GPS 坐标，格式 [lat, lon, z]
		# target_command: 高层驾驶命令的整数编号(如直行=0, 左转=1 等)
		target_gps, target_command = self.get_target_gps(input_data['gps'][1], compass)

		weather = self._weather_to_dict(self._world.get_weather())

		result = {
				'rgb': rgb,         # 前视图像 900x256 FOV:100
				'gps': gps,         # [纬度,经度]
				'speed': speed,     # 车速v_x和v_y
				'compass': compass, # 航向角
				'weather': weather, # 天气
				}
		next_wp, next_cmd = self._route_planner.run_step(self._get_position(result))  # 下一个参考路径点

		result['next_command'] = next_cmd.value
		result['x_target'] = next_wp[0]
		result['y_target'] = next_wp[1]
		result['bev_rgb']  = cv2.cvtColor(input_data['bev_rgb'][1][:, :, :3], cv2.COLOR_BGR2RGB)

		
		return result, obs_dict, birdview_obs['rendered'], target_gps, target_command, birdview_obs

	def im_render(self, render_dict):
		im_birdview = render_dict['rendered']
		h, w, c = im_birdview.shape
		im = np.zeros([h, w*2, c], dtype=np.uint8)
		im[:h, :w] = im_birdview

		action_str = np.array2string(render_dict['action'], precision=2, separator=',', suppress_small=True)

	
		txt_1 = f'a{action_str}'
		im = cv2.putText(im, txt_1, (3, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
					
		debug_texts = [ 
			'should_brake: ' + render_dict['should_brake'],
		]
		for i, txt in enumerate(debug_texts):
			im = cv2.putText(im, txt, (w, (i+2)*12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
		return im

	@torch.no_grad()
	def run_step(self, input_data, timestamp):  # 主循环 每帧执行一次
		if not self.initialized:
			self._init()

		self.step += 1

		if self.step < 20:

			control = carla.VehicleControl()
			control.steer = 0.0
			control.throttle = 0.0
			control.brake = 0.0
			self.last_control = control
			return control

		if self.step % 2 != 0:
			return self.last_control
		tick_data, policy_input, rendered, target_gps, target_command, veh_wal_info = self.tick(input_data, timestamp)

		gps = self._get_position(tick_data)

		near_node, near_command = self._waypoint_planner.run_step(gps)  # 最近的目标路径点 及其 高级指令
		far_node, far_command = self._command_planner.run_step(gps)     # 稍远的最近的目标路径点 及其 高级指令

		actions, values, log_probs, mu, sigma, features = self._policy.forward(
			policy_input, deterministic=True, clip_action=True)
		control = self.process_act(actions)  # [throttle, steer, brake]

		render_dict = {"rendered": rendered, "action": actions}

		# additional collision detection to enhance safety
		should_brake = self.collision_detect()
		only_ap_brake = True if (control.brake <= 0 and should_brake) else False  # 系统判断应该刹车，但车辆没有真正刹车
		if should_brake:
			control.steer = control.steer * 0.5
			control.throttle = 0.0
			control.brake = 1.0
		render_dict = {"rendered": rendered, "action": actions, "should_brake":str(should_brake),}
			
		render_img = self.im_render(render_dict)

		supervision_dict = {
			'action': np.array([control.throttle, control.steer, control.brake], dtype=np.float32),
			'value': values[0],         # 当前状态的价值
			'action_mu': mu[0],         # 当前策略的“最可能动作”
			'action_sigma': sigma[0],   # 动作分布的标准差 越大说明策略越不确定（或探索性强）
			'features': features[0],    # 策略网络将原始观测编码后的“语义状态表示”
			'speed': tick_data['speed'],# 车速,总线速度
			'target_gps': target_gps,   # 下一个导航点的 GPS 坐标，格式 [lat, lon, z]
			'target_command': target_command,  # 高层驾驶命令的整数编号(如直行=0, 左转=1 等)
			'should_brake': should_brake,      # 0 或者 1  表示是否有碰撞风险 0代表无碰撞风险 1代表有碰撞风险
			'only_ap_brake': only_ap_brake,    # False 或者 True  True代表:系统判断应该刹车，但车辆没有真正刹车  所以False是好的
		}
		if SAVE_PATH is not None and self.step % 10 == 0:
			self.save(near_node, far_node, near_command, far_command, tick_data, supervision_dict, render_img, should_brake, veh_wal_info)

		steer = control.steer
		control.steer = steer + 1e-2 * np.random.randn()
		self.last_control = control
		return control

	def collision_detect(self):
		# 判断当前环境中是否存在危险车辆或危险行人.
		# 返回一个布尔值,表示是否至少检测到了车辆或行人的碰撞风险.
		# 如果车辆或行人中有任何一个危险对象，返回 True，否则返回 False。
		actors = self._world.get_actors()

		vehicle = self._is_vehicle_hazard(actors.filter('*vehicle*'))
		walker = self._is_walker_hazard(actors.filter('*walker*'))


		self.is_vehicle_present = 1 if vehicle is not None else 0
		self.is_pedestrian_present = 1 if walker is not None else 0

		return any(x is not None for x in [vehicle, walker])

	def _is_walker_hazard(self, walkers_list):
		z = self._ego_vehicle.get_location().z
		p1 = _numpy(self._ego_vehicle.get_location())
		v1 = 10.0 * _orientation(self._ego_vehicle.get_transform().rotation.yaw)

		for walker in walkers_list:
			v2_hat = _orientation(walker.get_transform().rotation.yaw)
			s2 = np.linalg.norm(_numpy(walker.get_velocity()))

			if s2 < 0.05:
				v2_hat *= s2

			p2 = -3.0 * v2_hat + _numpy(walker.get_location())
			v2 = 8.0 * v2_hat

			collides, collision_point = get_collision(p1, v1, p2, v2)

			if collides:
				return walker

		return None

	def _is_vehicle_hazard(self, vehicle_list):
		z = self._ego_vehicle.get_location().z

		o1 = _orientation(self._ego_vehicle.get_transform().rotation.yaw)
		p1 = _numpy(self._ego_vehicle.get_location())
		s1 = max(10, 3.0 * np.linalg.norm(_numpy(self._ego_vehicle.get_velocity()))) # increases the threshold distance
		v1_hat = o1
		v1 = s1 * v1_hat

		for target_vehicle in vehicle_list:
			if target_vehicle.id == self._ego_vehicle.id:
				continue

			o2 = _orientation(target_vehicle.get_transform().rotation.yaw)
			p2 = _numpy(target_vehicle.get_location())
			s2 = max(5.0, 2.0 * np.linalg.norm(_numpy(target_vehicle.get_velocity())))
			v2_hat = o2
			v2 = s2 * v2_hat

			p2_p1 = p2 - p1
			distance = np.linalg.norm(p2_p1)
			p2_p1_hat = p2_p1 / (distance + 1e-4)

			angle_to_car = np.degrees(np.arccos(v1_hat.dot(p2_p1_hat)))
			angle_between_heading = np.degrees(np.arccos(o1.dot(o2)))

			# to consider -ve angles too
			angle_to_car = min(angle_to_car, 360.0 - angle_to_car)
			angle_between_heading = min(angle_between_heading, 360.0 - angle_between_heading)

			if angle_between_heading > 60.0 and not (angle_to_car < 15 and distance < s1):
				continue
			elif angle_to_car > 30.0:
				continue
			elif distance > s1:
				continue

			return target_vehicle

		return None

	def save(self, near_node, far_node, near_command, far_command, tick_data, supervision_dict, render_img, should_brake, veh_wal_info):
		frame = self.step // 10 - 2

		Image.fromarray(tick_data['rgb']).save(self.save_path / 'rgb' / ('%04d.png' % frame))
		Image.fromarray(tick_data['bev_rgb']).save(self.save_path / 'bev_rgb' / ('%04d.png' % frame))
		Image.fromarray(render_img).save(self.save_path / 'bev' / ('%04d.png' % frame))
		pos = self._get_position(tick_data)
		theta = tick_data['compass']
		speed = tick_data['speed']

		data = {
				'x': pos[0],
				'y': pos[1],
				'theta': theta,
				'speed': speed,
				'x_command_far': far_node[0],
				'y_command_far': far_node[1],
				'command_far': far_command.value,
				'x_command_near': near_node[0],
				'y_command_near': near_node[1],
				'command_near': near_command.value,
				'should_brake': should_brake,          # 0 或者 1  表示是否有碰撞风险 0代表无碰撞风险 1代表有碰撞风险
				'x_target': tick_data['x_target'],     # 下一个需要追踪的路径点的x坐标
				'y_target': tick_data['y_target'],     # 下一个需要追踪的路径点的y坐标
				'target_command': tick_data['next_command'],  # 下一个需要追踪的路径点所包含的高级导航指令
				'vehicles_info': veh_wal_info['vehicles_info'],
				'walker_info': veh_wal_info['walkers_info'],
				'route_pixels':veh_wal_info['route_pixels'],  # 全局路径的靠近自车前20个点的像素坐标
				'route_world': veh_wal_info['route_world'],   # 全局路径的靠近自车前20个点的carla世界坐标
				'route_relative_yaw_rad': veh_wal_info['route_relative_yaw_rad'],  # 全局路径的靠近自车前20个点相对于自车的航向角,负值代表向左,正值代表向右
				}
		outfile = open(self.save_path / 'measurements' / ('%04d.json' % frame), 'w')

		json.dump(data, outfile, indent=4)
		outfile.close()
		with open(self.save_path / 'supervision' / ('%04d.npy' % frame), 'wb') as f:
			np.save(f, supervision_dict)
		
			
	def get_target_gps(self, gps, compass):
		# target gps
		def gps_to_location(gps):
			lat, lon, z = gps
			lat = float(lat)
			lon = float(lon)
			z = float(z)

			location = carla.Location(z=z)
			xy =  (gps[:2] - self._command_planner.mean) * self._command_planner.scale
			location.x = xy[0]
			location.y = -xy[1]
			return location
		global_plan_gps = self._global_plan
		next_gps, _ = global_plan_gps[self.navigation_idx+1]
		next_gps = np.array([next_gps['lat'], next_gps['lon'], next_gps['z']])
		next_vec_in_global = gps_to_location(next_gps) - gps_to_location(gps)
		ref_rot_in_global = carla.Rotation(yaw=np.rad2deg(compass)-90.0)
		loc_in_ev = trans_utils.vec_global_to_ref(next_vec_in_global, ref_rot_in_global)

		if np.sqrt(loc_in_ev.x**2+loc_in_ev.y**2) < 12.0 and loc_in_ev.x < 0.0:
			self.navigation_idx += 1

		self.navigation_idx = min(self.navigation_idx, len(global_plan_gps)-2)

		_, road_option_0 = global_plan_gps[max(0, self.navigation_idx)]
		gps_point, road_option_1 = global_plan_gps[self.navigation_idx+1]
		gps_point = np.array([gps_point['lat'], gps_point['lon'], gps_point['z']])

		if (road_option_0 in [RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT]) \
				and (road_option_1 not in [RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT]):
			road_option = road_option_1
		else:
			road_option = road_option_0

		# gps_point: 下一个导航点的 GPS 坐标，格式 [lat, lon, z]
		# road_option.value: 高层驾驶命令的整数编号(如直行=0, 左转=1 等)
		return np.array(gps_point, dtype=np.float32), np.array([road_option.value], dtype=np.int8)


	def process_act(self, action):

		# acc, steer = action.astype(np.float64)
		acc = action[0][0]
		steer = action[0][1]
		if acc >= 0.0:
			throttle = acc
			brake = 0.0
		else:
			throttle = 0.0
			brake = np.abs(acc)

		throttle = np.clip(throttle, 0, 1)
		steer = np.clip(steer, -1, 1)
		brake = np.clip(brake, 0, 1)
		control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
		return control

	def _weather_to_dict(self, carla_weather):
		weather = {
			'cloudiness': carla_weather.cloudiness,
			'precipitation': carla_weather.precipitation,
			'precipitation_deposits': carla_weather.precipitation_deposits,
			'wind_intensity': carla_weather.wind_intensity,
			'sun_azimuth_angle': carla_weather.sun_azimuth_angle,
			'sun_altitude_angle': carla_weather.sun_altitude_angle,
			'fog_density': carla_weather.fog_density,
			'fog_distance': carla_weather.fog_distance,
			'wetness': carla_weather.wetness,
			'fog_falloff': carla_weather.fog_falloff,
		}

		return weather


	def _get_3d_bbs(self, max_distance=50):

		bounding_boxes = {
			"traffic_lights": [],
			"stop_signs": [],
			"vehicles": [],
			"pedestrians": []
		}

		bounding_boxes['traffic_lights'] = self._find_obstacle_3dbb('*traffic_light*', max_distance)
		bounding_boxes['stop_signs'] = self._find_obstacle_3dbb('*stop*', max_distance)
		bounding_boxes['vehicles'] = self._find_obstacle_3dbb('*vehicle*', max_distance)
		bounding_boxes['pedestrians'] = self._find_obstacle_3dbb('*walker*', max_distance)

		return bounding_boxes


	def _find_obstacle_3dbb(self, obstacle_type, max_distance=50):
		"""Returns a list of 3d bounding boxes of type obstacle_type.
		If the object does have a bounding box, this is returned. Otherwise a bb
		of size 0.5,0.5,2 is returned at the origin of the object.

		Args:
			obstacle_type (String): Regular expression
			max_distance (int, optional): max search distance. Returns all bbs in this radius. Defaults to 50.

		Returns:
			List: List of Boundingboxes
		"""        
		obst = list()
		
		_actors = self._world.get_actors()
		_obstacles = _actors.filter(obstacle_type)

		for _obstacle in _obstacles:    
			distance_to_car = _obstacle.get_transform().location.distance(self._ego_vehicle.get_location())

			if 0 < distance_to_car <= max_distance:
				
				if hasattr(_obstacle, 'bounding_box'): 
					loc = _obstacle.bounding_box.location
					_obstacle.get_transform().transform(loc)

					extent = _obstacle.bounding_box.extent
					_rotation_matrix = self.get_matrix(carla.Transform(carla.Location(0,0,0), _obstacle.get_transform().rotation))

					rotated_extent = np.squeeze(np.array((np.array([[extent.x, extent.y, extent.z, 1]]) @ _rotation_matrix)[:3]))

					bb = np.array([
						[loc.x, loc.y, loc.z],
						[rotated_extent[0], rotated_extent[1], rotated_extent[2]]
					])

				else:
					loc = _obstacle.get_transform().location
					bb = np.array([
						[loc.x, loc.y, loc.z],
						[0.5, 0.5, 2]
					])

				obst.append(bb)

		return obst

	def get_matrix(self, transform):
		"""
		Creates matrix from carla transform.
		"""

		rotation = transform.rotation
		location = transform.location
		c_y = np.cos(np.radians(rotation.yaw))
		s_y = np.sin(np.radians(rotation.yaw))
		c_r = np.cos(np.radians(rotation.roll))
		s_r = np.sin(np.radians(rotation.roll))
		c_p = np.cos(np.radians(rotation.pitch))
		s_p = np.sin(np.radians(rotation.pitch))
		matrix = np.matrix(np.identity(4))
		matrix[0, 3] = location.x
		matrix[1, 3] = location.y
		matrix[2, 3] = location.z
		matrix[0, 0] = c_p * c_y
		matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
		matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
		matrix[1, 0] = s_y * c_p
		matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
		matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
		matrix[2, 0] = s_p
		matrix[2, 1] = -c_p * s_r
		matrix[2, 2] = c_p * c_r
		return matrix

