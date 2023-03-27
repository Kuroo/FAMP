#!/bin/bash

# Algorithm settings
EPOCHS=11 # Outer optimization EPOCHS
ENVS_PER_PROCESS=1 # Envs to be sampled per process 
EPISODES=20 # Episodes per update step
LOOKAHEADS=0 # Inner updates to optimize for
GRAD_UPDATE_TYPE="single"

# Env settings
ENV_NAME="ant_maze_noreset"

# Discounts + Baseline settings
GAE_DISC=0.98
DICE_DISC=0
RET_DISC=0.99
BASELINE=linear

# Learning rate settings
LR_OUT=0.001
LR_IN=0.001
LEARN_PARAMS="inner"

# Policy settings
POLICY_TYPE="ttopt"
OPTS=3
HID_OPT=(64 64)
HID_SUBP=(64 64)
TERM_TIME=200

SEED=42
ENV=12


CHP_FREQ=10
RUN_NAME_PREFIX="eval"
RUN_DATA_DIR="../run_data"
CHP_DIR="$RUN_DATA_DIR/paper_experiments/ant_maze/oursft/pretrain"

CHP="${CHP_DIR}/seed${SEED}/checkpoints/epoch02700.tar"
python -u main.py --baseline $BASELINE --seed "$SEED" --env $ENV_NAME --run_name "env$ENV" \
--run_data_dir $RUN_DATA_DIR   --epochs $EPOCHS --envs_per_process $ENVS_PER_PROCESS --episodes $EPISODES \
--options $OPTS --gae_discount $GAE_DISC --lr_inner $LR_IN --lookaheads $LOOKAHEADS --dice_discount $DICE_DISC \
--grad_update_type $GRAD_UPDATE_TYPE --learn_params $LEARN_PARAMS --return_discount $RET_DISC \
--policy_type $POLICY_TYPE --hidden_sizes_option ${HID_OPT[*]} --term_time $TERM_TIME --load_chp $CHP \
--hidden_sizes_subpolicy ${HID_SUBP[*]} --chp_freq $CHP_FREQ --fixed_env $ENV --learn_lr_inner --save_trajs --visualize
