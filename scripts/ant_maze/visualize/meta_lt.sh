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
POLICY_TYPE="ltopt"
OPTS=3
HID_OPT=(64 64)
HID_TERM=(64 64)
HID_SUBP=(64 64) 


SEED=142
ENV=12

CHP_FREQ=10

CURREN_TIME=$(date "+%Y-%m-%d_%H-%M-%S")
RUN_NAME_PREFIX="eval"
RUN_DATA_DIR="../run_data"
CHP_DIR="$RUN_DATA_DIR/paper_experiments/ant_maze/famp/pretrain"

CHP="${CHP_DIR}/seed${SEED}/checkpoints/epoch02600.tar"
RUN_NAME=${RUN_NAME_PREFIX}_${GRAD_UPDATE_TYPE}_${ENV_NAME}_${OPTS}_${ENV}_${SEED}_${CURREN_TIME}
LOG_DIR="${RUN_DATA_DIR}/${RUN_NAME}/logs/"
python -u main.py --baseline $BASELINE --seed "$SEED" --env $ENV_NAME --run_name "$RUN_NAME" \
--run_data_dir $RUN_DATA_DIR   --epochs $EPOCHS --envs_per_process $ENVS_PER_PROCESS --episodes $EPISODES \
--options $OPTS --gae_discount $GAE_DISC --lr_inner $LR_IN --lookaheads $LOOKAHEADS --dice_discount $DICE_DISC \
--grad_update_type $GRAD_UPDATE_TYPE --learn_params $LEARN_PARAMS --return_discount $RET_DISC \
--policy_type $POLICY_TYPE --hidden_sizes_option ${HID_OPT[*]} --hidden_sizes_termination ${HID_TERM[*]} \
--hidden_sizes_subpolicy ${HID_SUBP[*]} --chp_freq $CHP_FREQ --load_chp $CHP --learn_lr_inner --fixed_env $ENV --visualize
