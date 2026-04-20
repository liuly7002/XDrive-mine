#!/bin/bash
export CARLA_ROOT=/home/liulei/ll/CARLA_0.9.10
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:leaderboard
export PYTHONPATH=$PYTHONPATH:leaderboard/team_code
export PYTHONPATH=$PYTHONPATH:scenario_runner

export PYTHONPATH=$PYTHONPATH:leaderboard
export PYTHONPATH=$PYTHONPATH:leaderboard/team_code
export PYTHONPATH=$PYTHONPATH:scenario_runner

export LEADERBOARD_ROOT=leaderboard
export CHALLENGE_TRACK_CODENAME=SENSORS
export PORT=2000
export TM_PORT=8000
export DEBUG_CHALLENGE=0
export REPETITIONS=1 # multiple evaluation runs
export RESUME=True


#export WEATHER=ClearNoon # ClearNoon, ClearSunset, CloudyNoon, CloudySunset, WetNoon, WetSunset, MidRainyNoon, MidRainSunset, WetCloudyNoon, WetCloudySunset, HardRainNoon, HardRainSunset, SoftRainNoon, SoftRainSunset
#export ROUTES=leaderboard/data/evaluation_routes/routes_town05_long.xml

# TCP evaluation
export ROUTES=leaderboard/data/evaluation_routes/routes_lav_valid.xml
export TEAM_AGENT=team_code/tcp_agent.py
#export TEAM_CONFIG=/home/liulei/ll/PPBEV_ll/A_result/version_41/epoch_013_valloss_0.6944.ckpt
#export TEAM_CONFIG=/home/liulei/ll/PPBEV_ll/A_result/version_41/best_epoch\=25-val_loss\=0.672.ckpt
#export TEAM_CONFIG=/home/liulei/ll/PPBEV_ll/A_result/version_42/epoch_011_valloss_0.9425.ckpt
#export TEAM_CONFIG=/home/liulei/ll/PPBEV_ll/A_result/version_42/epoch_031_valloss_0.8877.ckpt
#export TEAM_CONFIG=/home/liulei/ll/PPBEV_ll/A_result/version_42/epoch_045_valloss_0.8624.ckpt
#export TEAM_CONFIG=/home/liulei/ll/PPBEV_ll/A_result/version_42/epoch_055_valloss_0.8688.ckpt
#export TEAM_CONFIG=/home/liulei/ll/PPBEV_ll/A_result/version_42/epoch_047_valloss_0.8716.ckpt
#export TEAM_CONFIG=/home/liulei/ll/PPBEV_ll/A_result/version_44/epoch_025_valloss_0.5692.ckpt
#export TEAM_CONFIG=/home/liulei/ll/PPBEV_ll/A_result/version_44/epoch_031_valloss_0.5387.ckpt
#export TEAM_CONFIG=/home/liulei/ll/PPBEV_ll/A_result/version_45/best_epoch\=59-val_loss\=0.645.ckpt
#export TEAM_CONFIG=/home/liulei/ll/PPBEV_ll/A_result/version_45/epoch_049_valloss_0.6602.ckpt
#export TEAM_CONFIG=/home/liulei/ll/PPBEV_ll/A_result/version_45/epoch_041_valloss_0.6762.ckpt   # 最佳模型

# 消融实验——超视距
#export TEAM_CONFIG=/home/liulei/ll/PPBEV_ll/A_result/version_40/epoch_039_valloss_0.5760.ckpt        # 对照组  route_weight=0.001
# 消融实验——极点
#export TEAM_CONFIG=/home/liulei/ll/PPBEV_ll/A_result/version_37/best_epoch\=57-val_loss\=0.618.ckpt   # 25行18列
#export TEAM_CONFIG=/home/liulei/ll/PPBEV_ll/A_result/version_38/best_epoch\=31-val_loss\=0.595.ckpt   # 25行45列
#export TEAM_CONFIG=/home/liulei/ll/PPBEV_ll/A_result/version_46/epoch_037_valloss_0.9090.ckpt         # 25行27列
#export TEAM_CONFIG=/home/liulei/ll/PPBEV_ll/A_result/version_47/best_epoch\=43-val_loss\=0.677.ckpt   # 25行54列
#export TEAM_CONFIG=/home/liulei/ll/PPBEV_ll/A_result/version_48/best_epoch\=37-val_loss\=0.634.ckpt   # 20行36列
#export TEAM_CONFIG=/home/liulei/ll/PPBEV_ll/A_result/version_49/best_epoch\=47-val_loss\=0.606.ckpt   # 30行36列
export TEAM_CONFIG=/home/liulei/ll/PPBEV_ll/A_result/version_50/best_epoch\=39-val_loss\=0.557.ckpt    # 无极点表示


export CHECKPOINT_ENDPOINT=results/00x00/02/results_pp.json
export SCENARIOS=leaderboard/data/scenarios/all_towns_traffic_scenarios.json
export SAVE_PATH=results/00x00/02


python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
--scenarios=${SCENARIOS}  \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--resume=${RESUME} \
--port=${PORT} \
--trafficManagerPort=${TM_PORT}
#--weather=${WEATHER} #buat ganti2 weather


