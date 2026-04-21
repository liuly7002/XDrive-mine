from collections import deque
import numpy as np
import torch 
from torch import nn
from TCP.resnet import *
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import math
from TCP.decorder import PolarPointHead


class PIDController(object):
	def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
		self._K_P = K_P
		self._K_I = K_I
		self._K_D = K_D

		self._window = deque([0 for _ in range(n)], maxlen=n)
		self._max = 0.0
		self._min = 0.0

	def step(self, error):
		self._window.append(error)
		self._max = max(self._max, abs(error))
		self._min = -abs(self._max)

		if len(self._window) >= 2:
			integral = np.mean(self._window)
			derivative = (self._window[-1] - self._window[-2])
		else:
			integral = 0.0
			derivative = 0.0

		return self._K_P * error + self._K_I * integral + self._K_D * derivative

class TCP(nn.Module):

	def __init__(self, config):
		super().__init__()
		self.config = config

		self.turn_controller = PIDController(K_P=config.turn_KP, K_I=config.turn_KI, K_D=config.turn_KD, n=config.turn_n)
		self.speed_controller = PIDController(K_P=config.speed_KP, K_I=config.speed_KI, K_D=config.speed_KD, n=config.speed_n)

		self.use_route = True     # 是否将参考路径作为模态输入网络(超视距)
		self.use_pp = True        # 是否有极点表示
		self.opt_wp = False       # 是否对Trajectory生成进行优化
		self.only_traj = False    # 是否只预测Trajectory

		self.perception = resnet34(pretrained=True)

		if self.use_pp:
			# ==== 最好效果 ====
			self.polar_head = PolarPointHead(
				in_ch=512, mid_ch=128, out_h=32, out_w=32,
				use_block3=True,
				use_block4=True,
				use_res3=False,
				use_res4=False,
				use_axial=True,
				use_coord=True)

			"""
			消融实验  --> 开始
			"""
			# ==== Baseline ====
			# self.polar_head = PolarPointHead(
			# 	in_ch=512, mid_ch=128, out_h=25, out_w=36,
			# 	use_block3=False,  # baseline: 不使用 block3
			# 	use_block4=False,  # baseline: 不使用 block4
			# 	use_res3=False,    # baseline: 不使用残差3
			# 	use_res4=False,    # baseline: 不使用残差4
			# 	use_axial=False,   # baseline: 不使用 axial 注意力
			# 	use_coord=False    # baseline: 不使用坐标编码
			# )

			# ==== Ablation 1: 使用 block3 ====
			# self.polar_head = PolarPointHead(512, 128, 25, 36,
			#     use_block3=True, use_block4=False,
			#     use_res3=False, use_res4=False,
			#     use_axial=False, use_coord=False)

			# ==== Ablation 2: 使用 block3 + block4 ====
			# self.polar_head = PolarPointHead(512, 128, 25, 36,
			#     use_block3=True, use_block4=True,
			#     use_res3=False, use_res4=False,
			#     use_axial=False, use_coord=False)

			# ==== Ablation 3: 使用 block3 残差 ====
			# self.polar_head = PolarPointHead(512, 128, 25, 36,
			#     use_block3=True, use_block4=False,
			#     use_res3=True, use_res4=False,
			#     use_axial=False, use_coord=False)

			# ==== Ablation 4: 使用 block4 残差 ====
			# self.polar_head = PolarPointHead(512, 128, 25, 36,
			#     use_block3=True, use_block4=True,
			#     use_res3=False, use_res4=True,
			#     use_axial=False, use_coord=False)

			# ==== Ablation 5: 使用 axial 注意力 ====
			# self.polar_head = PolarPointHead(512, 128, 25, 36,
			#     use_block3=False, use_block4=False,
			#     use_res3=False, use_res4=False,
			#     use_axial=True, use_coord=False)

			# ==== Ablation 6: 使用坐标编码 ====
			# self.polar_head = PolarPointHead(512, 128, 25, 36,
			#     use_block3=False, use_block4=False,
			#     use_res3=False, use_res4=False,
			#     use_axial=False, use_coord=True)
			"""
			消融实验  --> 结束
			"""

		self.speed_branch = nn.Sequential(
							nn.Linear(1000, 256),
							nn.ReLU(inplace=True),
							nn.Linear(256, 256),
							nn.Dropout2d(p=0.5),
							nn.ReLU(inplace=True),
							nn.Linear(256, 1),
						)

		self.measurements = nn.Sequential(
							nn.Linear(1+2+6, 128),
							nn.ReLU(inplace=True),
							nn.Linear(128, 128),
							nn.ReLU(inplace=True),
						)

		if self.use_route:
			# 参考路径处理网络
			self.route_gru = nn.GRU(
				input_size=2,     # 每个路径点是 (x,y)
				hidden_size=128,  # 输出特征维度
				num_layers=1,
				batch_first=True,
				bidirectional=False
			)
			# 注意力池化
			self.route_attn = nn.Linear(128, 1)
			# 门控机制
			self.route_gate = nn.Linear(1000 + 128, 1)  # feature_emb + measurement_feature
			# 路径监督分支
			self.route_pred_head = nn.Sequential(
				nn.Linear(128, 128),
				nn.ReLU(inplace=True),
				nn.Linear(128, 2)  # 输出单个未来点 (x,y)
			)
			self.join_traj = nn.Sequential(
								nn.Linear(1000+128+128, 512),
								nn.ReLU(inplace=True),
								nn.Linear(512, 512),
								nn.ReLU(inplace=True),
								nn.Linear(512, 256),
								nn.ReLU(inplace=True),
							)
		else:
			self.join_traj = nn.Sequential(
								nn.Linear(1000+128, 512),
								nn.ReLU(inplace=True),
								nn.Linear(512, 512),
								nn.ReLU(inplace=True),
								nn.Linear(512, 256),
								nn.ReLU(inplace=True),
							)

		self.value_branch_traj = nn.Sequential(
					nn.Linear(256, 256),
					nn.ReLU(inplace=True),
					nn.Linear(256, 256),
					nn.Dropout2d(p=0.5),
					nn.ReLU(inplace=True),
					nn.Linear(256, 1),
				)

		self.output_traj = nn.Linear(256, 2)

		if self.opt_wp:
			self.time_emb = nn.Embedding(self.config.pred_len, 8)  # 小维度 (8)
			self.time_proj = nn.Linear(8, 256)

		self.decoder_traj = nn.GRUCell(input_size=4, hidden_size=256)

		if not self.only_traj:
			self.join_ctrl = nn.Sequential(
				nn.Linear(128 + 512, 512),
				nn.ReLU(inplace=True),
				nn.Linear(512, 512),
				nn.ReLU(inplace=True),
				nn.Linear(512, 256),
				nn.ReLU(inplace=True),
			)

			self.value_branch_ctrl = nn.Sequential(
				nn.Linear(256, 256),
				nn.ReLU(inplace=True),
				nn.Linear(256, 256),
				nn.Dropout2d(p=0.5),
				nn.ReLU(inplace=True),
				nn.Linear(256, 1),
			)

			# shared branches_neurons
			dim_out = 2

			self.policy_head = nn.Sequential(
					nn.Linear(256, 256),
					nn.ReLU(inplace=True),
					nn.Linear(256, 256),
					nn.Dropout2d(p=0.5),
					nn.ReLU(inplace=True),
				)
			self.decoder_ctrl = nn.GRUCell(input_size=256+4, hidden_size=256)
			self.output_ctrl = nn.Sequential(
					nn.Linear(256, 256),
					nn.ReLU(inplace=True),
					nn.Linear(256, 256),
					nn.ReLU(inplace=True),
				)
			self.dist_mu = nn.Sequential(nn.Linear(256, dim_out), nn.Softplus())
			self.dist_sigma = nn.Sequential(nn.Linear(256, dim_out), nn.Softplus())

			self.init_att = nn.Sequential(
					nn.Linear(128, 256),
					nn.ReLU(inplace=True),
					nn.Linear(256, 29*8),
					nn.Softmax(1)
				)

			self.wp_att = nn.Sequential(
					nn.Linear(256+256, 256),
					nn.ReLU(inplace=True),
					nn.Linear(256, 29*8),
					nn.Softmax(1)
				)

			self.merge = nn.Sequential(
					nn.Linear(512+256, 512),
					nn.ReLU(inplace=True),
					nn.Linear(512, 256),
				)
		

	def forward(self, img, state, target_point, gt_routepoints, route_len):

		outputs = {}

		feature_emb, cnn_feature = self.perception(img)   # cnn_feature.shape [batch_size,512,8,29]

		if self.use_pp:
			polar_logits = self.polar_head(cnn_feature)       # polar_logits.shape [B,1,25,36]
			outputs['pred_polar_logits'] = polar_logits               # [-,+]
			outputs['pred_polar_prob'] = torch.sigmoid(polar_logits)  # [0,1]

		outputs['pred_speed'] = self.speed_branch(feature_emb)
		measurement_feature = self.measurements(state)

		if self.use_route:
			gt_routepoints = gt_routepoints.float()  # [B, T_max, 2]
			if isinstance(route_len, list):
				route_len = torch.tensor(route_len, dtype=torch.long)
			route_len = route_len.cpu()    # 保证 route_len ≥ 1；若存在 0 长度样本，pack 会报错（必要时可把最小值 clamp 到 1，并把多出的点置零）。
			packed_input = nn.utils.rnn.pack_padded_sequence(gt_routepoints, lengths=route_len, batch_first=True, enforce_sorted=False)
			packed_output, _ = self.route_gru(packed_input)
			output_seq, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)  # [B, T_max, H]  对于短序列,超出 N_i 的部分会被 padding 置零

			# Attention pooling
			B, T_max, H = output_seq.shape
			logits = self.route_attn(output_seq).squeeze(-1)  # [B, T_max]  对每个隐状态打分(表示该时间步的重要性)
			idx = torch.arange(T_max, device=logits.device).unsqueeze(0).expand(B, -1)  # [B, T_max]
			valid_len = route_len.to(logits.device)
			pad_mask = idx >= valid_len.unsqueeze(1)  # True for padding
			logits = logits.masked_fill(pad_mask, float('-inf'))
			attn_weights = torch.softmax(logits, dim=1).unsqueeze(-1)  # [B, T_max, 1]
			route_feature = torch.sum(attn_weights * output_seq, dim=1)  # [B, H]

			# ---- 动态 α 调度 ----
			if hasattr(self, "global_step") and hasattr(self, "total_steps"):
				progress = min(1.0, self.global_step / self.total_steps)
				alpha = (progress ** 2)
			else:
				alpha = 0.0
			route_feature = alpha * route_feature

			# ---- LayerNorm 稳定 scale ----
			route_feature = F.layer_norm(route_feature, route_feature.shape[1:])
			# ---- Dropout 防止过拟合 ----
			p = 0.2
			if self.training and p > 0:
				keep = (torch.rand_like(route_feature) > p).float()
				route_feature = route_feature * keep / (1.0 - p)

			# 门控机制
			gate = torch.sigmoid(self.route_gate(torch.cat([feature_emb, measurement_feature], dim=1)))  # [B,1]
			route_feature = gate * route_feature

			# 监督分支输出
			outputs['route_wp'] = self.route_pred_head(route_feature)

			# 拼接特征
			j_traj = self.join_traj(torch.cat([feature_emb, measurement_feature, route_feature], dim=1))
		else:
			j_traj = self.join_traj(torch.cat([feature_emb, measurement_feature], 1))

		outputs['pred_value_traj'] = self.value_branch_traj(j_traj)
		outputs['pred_features_traj'] = j_traj

		output_wp = list()
		traj_hidden_state = list()
		z = j_traj
		x = torch.zeros(size=(z.shape[0], 2), dtype=z.dtype).type_as(z)  # [batch_size,2]
		for _ in range(self.config.pred_len):
			if self.opt_wp:
				t_emb = self.time_emb(torch.tensor(_, device=z.device, dtype=torch.long))  # [8]
				t_emb = self.time_proj(t_emb)  # [B, 256] 映射到隐状态维度
				z = z + t_emb  # 残差加法

			x_in = torch.cat([x, target_point], dim=1)  # [B,4]
			z = self.decoder_traj(x_in, z)
			traj_hidden_state.append(z)
			dx = self.output_traj(z)
			x = dx + x
			output_wp.append(x)
		pred_wp = torch.stack(output_wp, dim=1)
		outputs['pred_wp'] = pred_wp

		if not self.only_traj:
			traj_hidden_state = torch.stack(traj_hidden_state, dim=1)
			init_att = self.init_att(measurement_feature).view(-1, 1, 8, 29)
			feature_emb = torch.sum(cnn_feature*init_att, dim=(2, 3))
			j_ctrl = self.join_ctrl(torch.cat([feature_emb, measurement_feature], 1))
			outputs['pred_value_ctrl'] = self.value_branch_ctrl(j_ctrl)
			outputs['pred_features_ctrl'] = j_ctrl

			policy = self.policy_head(j_ctrl)
			outputs['mu_branches'] = self.dist_mu(policy)
			outputs['sigma_branches'] = self.dist_sigma(policy)

			x = j_ctrl
			mu = outputs['mu_branches']
			sigma = outputs['sigma_branches']
			future_feature, future_mu, future_sigma = [], [], []

			# initial hidden variable to GRU
			h = torch.zeros(size=(x.shape[0], 256), dtype=x.dtype).type_as(x)

			for _ in range(self.config.pred_len):
				x_in = torch.cat([x, mu, sigma], dim=1)
				h = self.decoder_ctrl(x_in, h)
				wp_att = self.wp_att(torch.cat([h, traj_hidden_state[:, _]], 1)).view(-1, 1, 8, 29)
				new_feature_emb = torch.sum(cnn_feature*wp_att, dim=(2, 3))
				merged_feature = self.merge(torch.cat([h, new_feature_emb], 1))
				dx = self.output_ctrl(merged_feature)
				x = dx + x

				policy = self.policy_head(x)
				mu = self.dist_mu(policy)
				sigma = self.dist_sigma(policy)
				future_feature.append(x)
				future_mu.append(mu)
				future_sigma.append(sigma)

			outputs['future_feature'] = future_feature
			outputs['future_mu'] = future_mu
			outputs['future_sigma'] = future_sigma

		return outputs

	def process_action(self, pred, command, speed, target_point):
		action = self._get_action_beta(pred['mu_branches'].view(1,2), pred['sigma_branches'].view(1,2))
		acc, steer = action.cpu().numpy()[0].astype(np.float64)
		if acc >= 0.0:
			throttle = acc
			brake = 0.0
		else:
			throttle = 0.0
			brake = np.abs(acc)

		throttle = np.clip(throttle, 0, 1)
		steer = np.clip(steer, -1, 1)
		brake = np.clip(brake, 0, 1)

		metadata = {
			'speed': float(speed.cpu().numpy().astype(np.float64)),
			'steer': float(steer),
			'throttle': float(throttle),
			'brake': float(brake),
			'command': command,
			'target_point': tuple(target_point[0].data.cpu().numpy().astype(np.float64)),
		}
		return steer, throttle, brake, metadata

	def _get_action_beta(self, alpha, beta):
		x = torch.zeros_like(alpha)
		x[:, 1] += 0.5
		mask1 = (alpha > 1) & (beta > 1)
		x[mask1] = (alpha[mask1]-1)/(alpha[mask1]+beta[mask1]-2)

		mask2 = (alpha <= 1) & (beta > 1)
		x[mask2] = 0.0

		mask3 = (alpha > 1) & (beta <= 1)
		x[mask3] = 1.0

		# mean
		mask4 = (alpha <= 1) & (beta <= 1)
		x[mask4] = alpha[mask4]/torch.clamp((alpha[mask4]+beta[mask4]), min=1e-5)

		x = x * 2 - 1

		return x

	def control_pid(self, waypoints, velocity, target):
		''' Predicts vehicle control with a PID controller.
		Args:
			waypoints (tensor): output of self.plan()
			velocity (tensor): speedometer input
		'''
		assert(waypoints.size(0)==1)
		waypoints = waypoints[0].data.cpu().numpy()
		target = target.squeeze().data.cpu().numpy()

		# flip y (forward is negative in our waypoints)
		waypoints[:,1] *= -1
		target[1] *= -1

		# iterate over vectors between predicted waypoints
		num_pairs = len(waypoints) - 1
		best_norm = 1e5
		desired_speed = 0
		aim = waypoints[0]
		for i in range(num_pairs):
			# magnitude of vectors, used for speed
			desired_speed += np.linalg.norm(
					waypoints[i+1] - waypoints[i]) * 2.0 / num_pairs

			# norm of vector midpoints, used for steering
			norm = np.linalg.norm((waypoints[i+1] + waypoints[i]) / 2.0)
			if abs(self.config.aim_dist-best_norm) > abs(self.config.aim_dist-norm):
				aim = waypoints[i]
				best_norm = norm

		aim_last = waypoints[-1] - waypoints[-2]

		angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
		angle_last = np.degrees(np.pi / 2 - np.arctan2(aim_last[1], aim_last[0])) / 90
		angle_target = np.degrees(np.pi / 2 - np.arctan2(target[1], target[0])) / 90

		# choice of point to aim for steering, removing outlier predictions
		# use target point if it has a smaller angle or if error is large
		# predicted point otherwise
		# (reduces noise in eg. straight roads, helps with sudden turn commands)
		use_target_to_aim = np.abs(angle_target) < np.abs(angle)
		use_target_to_aim = use_target_to_aim or (np.abs(angle_target-angle_last) > self.config.angle_thresh and target[1] < self.config.dist_thresh)
		if use_target_to_aim:
			angle_final = angle_target
		else:
			angle_final = angle

		steer = self.turn_controller.step(angle_final)
		steer = np.clip(steer, -1.0, 1.0)

		speed = velocity[0].data.cpu().numpy()
		brake = desired_speed < self.config.brake_speed or (speed / desired_speed) > self.config.brake_ratio

		delta = np.clip(desired_speed - speed, 0.0, self.config.clip_delta)
		throttle = self.speed_controller.step(delta)
		throttle = np.clip(throttle, 0.0, self.config.max_throttle)
		throttle = throttle if not brake else 0.0

		metadata = {
			'speed': float(speed.astype(np.float64)),
			'steer': float(steer),
			'throttle': float(throttle),
			'brake': float(brake),
			'wp_4': tuple(waypoints[3].astype(np.float64)),
			'wp_3': tuple(waypoints[2].astype(np.float64)),
			'wp_2': tuple(waypoints[1].astype(np.float64)),
			'wp_1': tuple(waypoints[0].astype(np.float64)),
			'aim': tuple(aim.astype(np.float64)),
			'target': tuple(target.astype(np.float64)),
			'desired_speed': float(desired_speed.astype(np.float64)),
			'angle': float(angle.astype(np.float64)),
			'angle_last': float(angle_last.astype(np.float64)),
			'angle_target': float(angle_target.astype(np.float64)),
			'angle_final': float(angle_final.astype(np.float64)),
			'delta': float(delta.astype(np.float64)),
		}

		return steer, throttle, brake, metadata

	def get_action(self, mu, sigma):
		if mu.dim() == 1:             # 如果传入的mu是一维的(例如[2]),则扩展为2维(例如[1,2])
			mu = mu.unsqueeze(0)
			sigma = sigma.unsqueeze(0)

		action = self._get_action_beta(mu, sigma)
		acc, steer = action[:, 0], action[:, 1]  # acc: 加速度, steer: 转向角  shape:[batch_size]

		throttle = torch.clamp(acc, min=0.0, max=1.0)
		steer = torch.clamp(steer, -1.0, 1.0)
		brake = torch.clamp(-acc, min=0.0, max=1.0)

		return throttle, steer, brake  # throttle.shape=[batch_size]  steer.shape=[batch_size]  brake.shape=[batch_size]
