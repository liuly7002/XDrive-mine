import argparse
import os
from collections import OrderedDict

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.distributions import Beta

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin

from TCP.model import TCP
from TCP.data import CARLA_Data
from TCP.config import GlobalConfig

from visualize_attention import VISUALIZE_ATTENTION, overlay_attention_on_img
# import matplotlib.pyplot as plt


class TCP_planner(pl.LightningModule):
	def __init__(self, config, lr):
		super().__init__()
		self.lr = lr
		self.config = config
		self.model = TCP(config)
		self._load_weight()
		self.load_ckpt()

	def _load_weight(self):
		rl_state_dict = torch.load(self.config.rl_ckpt, map_location='cpu')['policy_state_dict']
		self._load_state_dict(self.model.value_branch_traj, rl_state_dict, 'value_head')
		if not self.model.only_traj:
			self._load_state_dict(self.model.value_branch_ctrl, rl_state_dict, 'value_head')
			self._load_state_dict(self.model.dist_mu, rl_state_dict, 'dist_mu')
			self._load_state_dict(self.model.dist_sigma, rl_state_dict, 'dist_sigma')

	def _load_state_dict(self, il_net, rl_state_dict, key_word):
		rl_keys = [k for k in rl_state_dict.keys() if key_word in k]
		il_keys = il_net.state_dict().keys()
		assert len(rl_keys) == len(il_net.state_dict().keys()), f'mismatch number of layers loading {key_word}'
		new_state_dict = OrderedDict()
		for k_il, k_rl in zip(il_keys, rl_keys):
			new_state_dict[k_il] = rl_state_dict[k_rl]
		il_net.load_state_dict(new_state_dict)

	def load_ckpt(self):
		ckpt = torch.load('./pretrain_weight.ckpt')["state_dict"]
		new_state_dict = OrderedDict()
		for key, value in ckpt.items():
			new_key = key.replace("model.", "")
			new_state_dict[new_key] = value

		# === 修改部分：过滤掉 shape 不匹配的参数 ===
		model_dict = self.model.state_dict()
		filtered_dict = {}
		for k, v in new_state_dict.items():
			if k in model_dict:
				if model_dict[k].shape == v.shape:
					filtered_dict[k] = v
				else:
					print(f"⚠️ Skip loading '{k}' due to shape mismatch: {v.shape} -> {model_dict[k].shape}")
			else:
				print(f"⚠️ Skip loading '{k}' (not found in model)")
		model_dict.update(filtered_dict)
		self.model.load_state_dict(model_dict)
		print(f"✅ Successfully loaded {len(filtered_dict)} matching layers from pretrained checkpoint.")

	def forward(self, batch):
		pass

	def compute_metrics(self, pred_wp, gt_wp, mr_threshold=1.0):
		"""
        统一计算轨迹预测(wp)指标：ADE, FDE, MR

        参数:
            pred_wp: [B, T_pred, 2] 预测轨迹
            gt_wp:   [B, T_pred, 2] 真实轨迹
            mr_threshold: float, MR 阈值（单位与坐标系一致）

        返回:
            ade_mean: float, batch 内平均 ADE
            fde_mean: float, batch 内平均 FDE
            mr: float, batch 内 Miss Rate
        """
		diff = pred_wp - gt_wp    # [B, T_pred, 2]   误差
		dist = diff.norm(dim=-1)  # [B, T_pred]      误差均一化

		# ADE 和 FDE
		ade = dist.mean(dim=1)  # [B] 每条轨迹的平均误差
		fde = dist[:, -1]       # [B] 每条轨迹最后一步误差

		ade_mean = ade.mean().item()
		fde_mean = fde.mean().item()

		# MR
		misses = (fde > mr_threshold).float()
		mr = misses.mean().item()

		return ade_mean, fde_mean, mr

	def training_step(self, batch, batch_idx):

		self.model.global_step = self.global_step
		if self.trainer is not None:
			try:
				total_steps = self.trainer.num_training_batches * max(1, self.trainer.max_epochs)
			except:
				total_steps = 1
			self.model.total_steps = total_steps
		progress = min(1.0, self.model.global_step / self.model.total_steps)
		alpha = 0.1 + 0.9 * progress  # 0.1 -> 1.0
		self.log('train_alpha', alpha)

		front_img = batch['front_img']

		speed = batch['speed'].to(dtype=torch.float32).view(-1,1) / 12.
		target_point = batch['target_point'].to(dtype=torch.float32)
		command = batch['target_command']
		state = torch.cat([speed, target_point, command], 1)

		value = batch['value'].view(-1,1)
		feature = batch['feature']

		gt_waypoints = batch['waypoints']       # 自车未来4帧的位置, 自车坐标系下的位置

		gt_routepoints = batch['routepoints']   # 参考路径20个点(存在padding的点)在自车坐标系下的位置
		route_len = batch['route_len']          # 真实的路径长度(除去padding的点)

		gt_polarpoints = batch['polarpoints']   # 极点 Ground Truth

		pred = self.model(front_img, state, target_point, gt_routepoints, route_len)

		if self.model.use_pp:
			bce = torch.nn.BCEWithLogitsLoss()(pred['pred_polar_logits'], gt_polarpoints)    # 不作为损失！！！
			mse = F.mse_loss(pred['pred_polar_prob'], gt_polarpoints)
			l1 = F.l1_loss(pred['pred_polar_prob'], gt_polarpoints)
			pp_loss = self.config.pp_weight * ( 0.0 * bce + 1.0 * mse + 0.5 * l1 )           # 调参空间: (1.0, 0.5, 0.5) 只是起点
			self.log('train_pp_loss', pp_loss.item())

		speed_loss = F.l1_loss(pred['pred_speed'], speed) * self.config.speed_weight
		self.log('train_speed_loss', speed_loss.item())

		wp_loss = F.l1_loss(pred['pred_wp'], gt_waypoints, reduction='none').mean()
		self.log('train_wp_loss', wp_loss.item())

		if self.model.only_traj:
			value_loss = F.mse_loss(pred['pred_value_traj'], value) * self.config.value_weight
			self.log('train_value_loss', value_loss.item())

			feature_loss = F.mse_loss(pred['pred_features_traj'], feature) * self.config.features_weight
			self.log('train_feature_loss', feature_loss.item())

			if self.model.use_route:
				batch_size = gt_routepoints.size(0)
				device = gt_routepoints.device
				idx = route_len - 1  # [B]
				batch_idx = torch.arange(batch_size, device=device)
				# 取每条路径的最后一个有效点
				gt_future_point = gt_routepoints[batch_idx, idx]  # [B, 2]
				route_loss = self.config.route_weight * F.l1_loss(pred['route_wp'], gt_future_point)
				self.log('train_route_loss', route_loss.item())

				if self.model.use_pp:
					loss = speed_loss + value_loss + feature_loss + wp_loss + route_loss + pp_loss
				else:
					loss = speed_loss + value_loss + feature_loss + wp_loss + route_loss
			else:
				if self.model.use_pp:
					loss = speed_loss + value_loss + feature_loss + wp_loss + pp_loss
				else:
					loss = speed_loss + value_loss + feature_loss + wp_loss
		else:
			value_loss = (F.mse_loss(pred['pred_value_traj'], value) + F.mse_loss(pred['pred_value_ctrl'],value)) * self.config.value_weight
			self.log('train_value_loss', value_loss.item())

			feature_loss = (F.mse_loss(pred['pred_features_traj'], feature) + F.mse_loss(pred['pred_features_ctrl'],feature)) * self.config.features_weight
			self.log('train_feature_loss', feature_loss.item())

			dist_sup = Beta(batch['action_mu'], batch['action_sigma'])
			dist_pred = Beta(pred['mu_branches'], pred['sigma_branches'])
			kl_div = torch.distributions.kl_divergence(dist_sup, dist_pred)
			action_loss = torch.mean(kl_div[:, 0]) *0.5 + torch.mean(kl_div[:, 1]) *0.5
			self.log('train_action_loss', action_loss.item())

			future_feature_loss = 0
			future_action_loss = 0
			for i in range(self.config.pred_len):
				dist_sup = Beta(batch['future_action_mu'][i], batch['future_action_sigma'][i])
				dist_pred = Beta(pred['future_mu'][i], pred['future_sigma'][i])   # 这里有问题
				kl_div = torch.distributions.kl_divergence(dist_sup, dist_pred)
				future_action_loss += torch.mean(kl_div[:, 0]) *0.5 + torch.mean(kl_div[:, 1]) *0.5
				future_feature_loss += F.mse_loss(pred['future_feature'][i], batch['future_feature'][i]) * self.config.features_weight
			future_feature_loss /= self.config.pred_len
			future_action_loss /= self.config.pred_len
			self.log('train_future_feature_loss', future_feature_loss.item())
			self.log('train_future_action_loss', future_action_loss.item())

			if self.model.use_route:
				batch_size = gt_routepoints.size(0)
				device = gt_routepoints.device
				idx = route_len - 1  # [B]
				batch_idx = torch.arange(batch_size, device=device)
				# 取每条路径的最后一个有效点
				gt_future_point = gt_routepoints[batch_idx, idx]  # [B, 2]
				route_loss = self.config.route_weight * F.l1_loss(pred['route_wp'], gt_future_point)
				self.log('train_route_loss', route_loss.item())

				if self.model.use_pp:
					loss = action_loss + speed_loss + value_loss + feature_loss + wp_loss + future_feature_loss + future_action_loss + route_loss + pp_loss
				else:
					loss = action_loss + speed_loss + value_loss + feature_loss + wp_loss + future_feature_loss + future_action_loss + route_loss
			else:
				if self.model.use_pp:
					loss = action_loss + speed_loss + value_loss + feature_loss + wp_loss + future_feature_loss + future_action_loss + pp_loss
				else:
					loss = action_loss + speed_loss + value_loss + feature_loss + wp_loss + future_feature_loss + future_action_loss

		return loss

	def configure_optimizers(self):
		optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-7)
		lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 30, 0.5)
		return [optimizer], [lr_scheduler]

	def validation_step(self, batch, batch_idx):

		front_img = batch['front_img']
		speed = batch['speed'].to(dtype=torch.float32).view(-1,1) / 12.
		target_point = batch['target_point'].to(dtype=torch.float32)
		command = batch['target_command']
		state = torch.cat([speed, target_point, command], 1)
		value = batch['value'].view(-1,1)
		feature = batch['feature']
		gt_waypoints = batch['waypoints']
		gt_routepoints = batch['routepoints']   # 参考路径20个点在自车坐标系下的位置
		route_len = batch['route_len']          # 参考路径中真实有效的路径点数量
		gt_polarpoints = batch['polarpoints']   # 极点 Ground Truth

		pred = self.model(front_img, state, target_point, gt_routepoints, route_len)

		# =========================== 验证步可视化注意力 ===========================
		if self.model.use_pp:
			if VISUALIZE_ATTENTION and batch_idx % 20 == 0:
				attn_maps = self.model.polar_head.last_attn_maps
				save_dir = "attention_vis"
				os.makedirs(save_dir, exist_ok=True)

				row_attn = attn_maps['row']
				col_attn = attn_maps['col']

				# 融合注意力
				if row_attn is not None and col_attn is not None:
					fused = (row_attn + col_attn) / 2.0  # [B,H,W]
				else:
					fused = row_attn if row_attn is not None else col_attn

				if fused is not None:
					idx = 0  # 只保存第一张
					img_np = front_img[idx].detach().cpu().permute(1, 2, 0).numpy()
					amap_np = fused[idx].detach().cpu().numpy()
					overlay = overlay_attention_on_img(img_np, amap_np)

					# 保存 fused 注意力图
					save_path = os.path.join(
						save_dir,
						f"step{self.global_step}_batch{batch_idx}_fused.png"
					)
					from matplotlib import pyplot as plt
					fig, axes = plt.subplots(1, 3, figsize=(12, 4))
					axes[0].imshow(img_np)
					axes[0].set_title("Original")
					axes[0].axis("off")
					axes[1].imshow(amap_np, cmap="jet")
					axes[1].set_title("Fused Attention Map")
					axes[1].axis("off")
					axes[2].imshow(overlay)
					axes[2].set_title("Overlay")
					axes[2].axis("off")

					plt.savefig(save_path, bbox_inches="tight", dpi=150)
					plt.close(fig)

		# =========================== 极点验证指标: BCE(不作为验证指标) + MSE + L1 + 线性相关性===========================
		if self.model.use_pp:
			pp_bce = torch.nn.BCEWithLogitsLoss()(pred['pred_polar_logits'], gt_polarpoints)
			pp_mse = F.mse_loss(pred['pred_polar_prob'], gt_polarpoints)
			pp_l1 = F.l1_loss(pred['pred_polar_prob'], gt_polarpoints)
			self.log('val_pp_bce', pp_bce.item())                  # 越小表示预测越接近 GT  不作为指标！！！
			self.log('val_pp_mse', pp_mse.item(), sync_dist=True)   # 越小表示预测越接近 GT
			self.log('val_pp_l1', pp_l1.item(), sync_dist=True)     # 越小表示预测越接近 GT
			gt_flat = gt_polarpoints.view(gt_polarpoints.size(0), -1)
			pred_flat = pred['pred_polar_prob'].view(pred['pred_polar_prob'].size(0), -1)
			mean_gt = gt_flat.mean(dim=1, keepdim=True)
			mean_pred = pred_flat.mean(dim=1, keepdim=True)
			cov = ((gt_flat - mean_gt) * (pred_flat - mean_pred)).sum(dim=1)
			std_gt = torch.sqrt(((gt_flat - mean_gt) ** 2).sum(dim=1) + 1e-6)
			std_pred = torch.sqrt(((pred_flat - mean_pred) ** 2).sum(dim=1) + 1e-6)
			pp_corr = (cov / (std_gt * std_pred)).mean()
			self.log('val_pp_corr', pp_corr.item(), sync_dist=True)    # 皮尔逊相关系数(Pearson Correlation Coefficient), 越接近 1 表示预测与 GT 越相关

		# =========================== waypoint验证指标: ADE FDE MR ===========================
		ade_normal, fde_normal, mr_normal = self.compute_metrics(pred['pred_wp'], gt_waypoints, mr_threshold=1.0)
		self.log('val_ade_normal', ade_normal, sync_dist=True)
		self.log('val_fde_normal', fde_normal, sync_dist=True)
		self.log('val_mr_normal', mr_normal, sync_dist=True)
		wp_loss = F.l1_loss(pred['pred_wp'], gt_waypoints, reduction='none').mean()
		self.log('val_wp_loss', wp_loss.item(), sync_dist=True)

		# =========================== 自车speed验证指标: L1 loss ===========================
		speed_loss = F.l1_loss(pred['pred_speed'], speed) * self.config.speed_weight
		self.log('val_speed_loss', speed_loss.item(), sync_dist=True)

		if self.model.only_traj:
			value_loss = F.mse_loss(pred['pred_value_traj'], value) * self.config.value_weight
			self.log('val_value_loss', value_loss.item(), sync_dist=True)

			feature_loss = F.mse_loss(pred['pred_features_traj'], feature) * self.config.features_weight
			self.log('val_feature_loss', feature_loss.item(), sync_dist=True)

			if self.model.use_route:
				batch_size = gt_routepoints.size(0)
				device = gt_routepoints.device
				idx = route_len - 1  # [B]
				batch_idx = torch.arange(batch_size, device=device)
				# 取每条路径的最后一个有效点
				gt_future_point = gt_routepoints[batch_idx, idx]  # [B, 2]
				route_loss = F.l1_loss(pred['route_wp'], gt_future_point)
				self.log('val_route_loss', route_loss.item(), sync_dist=True)

				if self.model.use_pp:
					val_loss = wp_loss + self.config.route_weight * route_loss + self.config.pp_weight * ( 0.0 * pp_bce + 1.0 * pp_mse + 0.5 * pp_l1 )
				else:
					val_loss = wp_loss + self.config.route_weight * route_loss
			else:
				if self.model.use_pp:
					val_loss = wp_loss + self.config.pp_weight * ( 0.0 * pp_bce + 1.0 * pp_mse + 0.5 * pp_l1 )
				else:
					val_loss = wp_loss

			self.log('val_loss', val_loss.item(), sync_dist=True)
		else:
			value_loss = (F.mse_loss(pred['pred_value_traj'], value) + F.mse_loss(pred['pred_value_ctrl'],value)) * self.config.value_weight
			self.log('val_value_loss', value_loss.item(), sync_dist=True)

			feature_loss = (F.mse_loss(pred['pred_features_traj'], feature) + F.mse_loss(pred['pred_features_ctrl'],feature)) * self.config.features_weight
			self.log('val_feature_loss', feature_loss.item(), sync_dist=True)

			# =========================== Action(当前帧)验证指标 ===========================
			dist_sup = Beta(batch['action_mu'], batch['action_sigma'])
			dist_pred = Beta(pred['mu_branches'], pred['sigma_branches'])
			kl_div = torch.distributions.kl_divergence(dist_sup, dist_pred)
			action_loss = torch.mean(kl_div[:, 0]) *0.5 + torch.mean(kl_div[:, 1]) *0.5
			self.log("val_action_loss", action_loss.item(), sync_dist=True)

			# =========================== 油门、转向、刹车(当前帧)验证指标: L1 loss ===========================
			B = batch['action_mu'].shape[0]
			batch_steer_l1 = 0
			batch_brake_l1 = 0
			batch_throttle_l1 = 0
			for i in range(B):
				throttle, steer, brake = self.model.get_action(pred['mu_branches'][i], pred['sigma_branches'][i])
				batch_throttle_l1 += torch.abs(throttle-batch['action'][i][0])
				batch_steer_l1 += torch.abs(steer-batch['action'][i][1])
				batch_brake_l1 += torch.abs(brake-batch['action'][i][2])
			batch_throttle_l1 /= B  # 油门
			batch_steer_l1 /= B     # 转向
			batch_brake_l1 /= B     # 刹车
			self.log("val_throttle", batch_throttle_l1, sync_dist=True)
			self.log("val_steer", batch_steer_l1, sync_dist=True)
			self.log("val_brake", batch_brake_l1, sync_dist=True)

			# =========================== Action(未来帧)验证指标 ===========================
			future_feature_loss = 0
			future_action_loss = 0
			for i in range(self.config.pred_len-1):
				dist_sup = Beta(batch['future_action_mu'][i], batch['future_action_sigma'][i])
				dist_pred = Beta(pred['future_mu'][i], pred['future_sigma'][i])
				kl_div = torch.distributions.kl_divergence(dist_sup, dist_pred)
				future_action_loss += torch.mean(kl_div[:, 0]) *0.5 + torch.mean(kl_div[:, 1]) *0.5
				future_feature_loss += F.mse_loss(pred['future_feature'][i], batch['future_feature'][i]) * self.config.features_weight
			future_feature_loss /= self.config.pred_len
			future_action_loss /= self.config.pred_len
			self.log('val_future_feature_loss', future_feature_loss.item(), sync_dist=True)
			self.log("val_future_action_loss", future_action_loss, sync_dist=True)

			# =========================== 油门、转向、刹车(未来帧)验证指标: L1误差 ===========================
			future_throttle_l1 = future_steer_l1 = future_brake_l1 = 0
			for t in range(self.config.pred_len - 1):
				# 将未来动作分布转为具体动作
				throttle_pred, steer_pred, brake_pred = self.model.get_action(pred['future_mu'][t], pred['future_sigma'][t])
				throttle_gt, steer_gt, brake_gt = self.model.get_action(batch['future_action_mu'][t], batch['future_action_sigma'][t])
				# 计算L1误差
				future_throttle_l1 += torch.mean(torch.abs(throttle_pred - throttle_gt))
				future_steer_l1    += torch.mean(torch.abs(steer_pred - steer_gt))
				future_brake_l1    += torch.mean(torch.abs(brake_pred - brake_gt))
			# 对未来所有预测步求平均
			future_throttle_l1 /= (self.config.pred_len - 1)
			future_steer_l1 /= (self.config.pred_len - 1)
			future_brake_l1 /= (self.config.pred_len - 1)
			self.log("val_future_throttle", future_throttle_l1, sync_dist=True)
			self.log("val_future_steer", future_steer_l1, sync_dist=True)
			self.log("val_future_brake", future_brake_l1, sync_dist=True)

			if self.model.use_route:
				batch_size = gt_routepoints.size(0)
				device = gt_routepoints.device
				idx = route_len - 1  # [B]
				batch_idx = torch.arange(batch_size, device=device)
				# 取每条路径的最后一个有效点
				gt_future_point = gt_routepoints[batch_idx, idx]  # [B, 2]
				route_loss = F.l1_loss(pred['route_wp'], gt_future_point)
				self.log('val_route_loss', route_loss.item(), sync_dist=True)

				if self.model.use_pp:
					val_loss = wp_loss + batch_throttle_l1 + 5 * batch_steer_l1 + batch_brake_l1 + self.config.route_weight * route_loss + self.config.pp_weight * ( 0.0 * pp_bce + 1.0 * pp_mse + 0.5 * pp_l1 )
				else:
					val_loss = wp_loss + batch_throttle_l1 + 5 * batch_steer_l1 + batch_brake_l1 + self.config.route_weight * route_loss
			else:
				if self.model.use_pp:
					val_loss = wp_loss + batch_throttle_l1 + 5 * batch_steer_l1 + batch_brake_l1 + self.config.pp_weight * ( 0.0 * pp_bce + 1.0 * pp_mse + 0.5 * pp_l1 )
				else:
					val_loss = wp_loss + batch_throttle_l1 + 5 * batch_steer_l1 + batch_brake_l1

			self.log('val_loss', val_loss.item(), sync_dist=True)

		return {'val_loss': val_loss.detach()}

	def validation_epoch_end(self, outputs):
		epoch = self.current_epoch

		# === 从 outputs 汇总 val_loss ===
		val_losses = [x['val_loss'].detach().cpu().item() for x in outputs if 'val_loss' in x]
		mean_val_loss = sum(val_losses) / len(val_losses) if len(val_losses) > 0 else 0.0

		# === 保存 ckpt ===
		save_dir = "checkpoints"
		os.makedirs(save_dir, exist_ok=True)
		save_path = os.path.join(save_dir, f"epoch_{epoch:03d}_valloss_{mean_val_loss:.4f}.ckpt")

		ckpt = {
			'epoch': epoch,
			'state_dict': self.state_dict(),
			'optimizer_states': [opt.state_dict() for opt in self.trainer.optimizers],
			'val_loss': mean_val_loss,
			'config': getattr(self, "config", None)
		}

		torch.save(ckpt, save_path)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument('--id', type=str, default='TCP', help='Unique experiment identifier.')
	parser.add_argument('--epochs', type=int, default=60, help='Number of train epochs.')
	parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
	parser.add_argument('--val_every', type=int, default=2, help='Validation frequency (epochs).')
	parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
	parser.add_argument('--logdir', type=str, default='log', help='Directory to log data to.')
	parser.add_argument('--gpus', type=int, default=1, help='number of gpus')

	args = parser.parse_args()
	args.logdir = os.path.join(args.logdir, args.id)

	# Config
	config = GlobalConfig()

	# Data
	train_set = CARLA_Data(root=config.root_dir_all, data_folders=config.train_data, img_aug = config.img_aug)
	print(len(train_set))
	val_set = CARLA_Data(root=config.root_dir_all, data_folders=config.val_data,)
	print(len(val_set))

	dataloader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8)
	dataloader_val = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=8)

	TCP_model = TCP_planner(config, args.lr)

	checkpoint_callback = ModelCheckpoint(save_weights_only=False, mode="min", monitor="val_loss", save_top_k=2, save_last=True,
											dirpath=args.logdir, filename="best_{epoch:02d}-{val_loss:.3f}")
	checkpoint_callback.CHECKPOINT_NAME_LAST = "{epoch}-last"
	trainer = pl.Trainer.from_argparse_args(args,
											default_root_dir=args.logdir,
											gpus = args.gpus,
											accelerator='ddp',
											sync_batchnorm=True,
											plugins=DDPPlugin(find_unused_parameters=False),
											profiler='simple',
											benchmark=True,
											log_every_n_steps=1,
											flush_logs_every_n_steps=5,
											callbacks=[checkpoint_callback,
														],
											check_val_every_n_epoch = args.val_every,
											max_epochs = args.epochs
											)

	trainer.fit(TCP_model, dataloader_train, dataloader_val)