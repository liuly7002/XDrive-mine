#!/bin/bash
export CARLA_ROOT=/home/a/carla0.9.10_package/CARLA_0.9.10-dirty
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
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
export DATA_COLLECTION=True


# Roach data collection
#export ROUTES=leaderboard/data/TCP_training_routes/routes_town05_val.xml                  # 修改1: 路径
export ROUTES=/home/a/XDrive-mine-main/leaderboard/data/11.xml                # 修改1: 路径

export TEAM_AGENT=team_code/roach_ap_agent.py
export TEAM_CONFIG=roach/config/config_agent.yaml
#export CHECKPOINT_ENDPOINT=data_collect_routes_town05_val_00_results.json                    # 修改2: 路径对应的结果文件  _original
export CHECKPOINT_ENDPOINT=data/data_collect_routes_kuangshan_00_results.json                    # 修改2: 路径对应的结果文件  _original
export SCENARIOS=leaderboard/data/ceshi.json
#export SAVE_PATH=data/town05_val_00/                                                         # 修改3: 收集数据存放的位置  _original
export SAVE_PATH=data/kuangshan_00/                                                         # 修改3: 收集数据存放的位置  _original
export DATAGEN=1



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


