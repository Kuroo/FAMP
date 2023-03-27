#!/bin/bash

# Algorithm settings
EPOCHS=30 # Outer optimization EPOCHS
ENVS_PER_PROCESS=1 # Envs to be sampled per process (for maxspeed on cluster=1)
EPISODES=10 # Episodes per update step
LOOKAHEADS=0 # Inner updates to optimize for
GRAD_UPDATE_TYPE="single"

# Env settings
ENV_NAME="taxi"

# Discounts + Baseline settings
GAE_DISC=0.98
DICE_DISC=0
RET_DISC=0.95
BASELINE=linear

# Learning rate settings
LR_OUT=0.01
LR_IN=10.0
LEARN_PARAMS="inner"

# Policy settings
POLICY_TYPE="ltopt"
OPTS=4

#Others
CHP_FREQ=30
PLOT_FREQ=5000

SEED=42
ENV=26

CURREN_TIME=$(date "+%Y-%m-%d_%H-%M-%S")
RUN_DIR="../run_data/paper_experiments/taxi/famp"

CHP="${RUN_DIR}/train/seed${SEED}/checkpoints/epoch2000.tar"
RUN_NAME="env${ENV}"
LOG_DIR="${RUN_DIR}/eval/seed${SEED}"
python -u main.py --baseline $BASELINE --seed "$SEED" --env $ENV_NAME --run_name "$RUN_NAME" \
--run_data_dir ${LOG_DIR} --epochs $EPOCHS --envs_per_process $ENVS_PER_PROCESS --episodes $EPISODES --options $OPTS \
--fixed_env $ENV --gae_discount $GAE_DISC --lr_inner $LR_IN --lookaheads $LOOKAHEADS --dice_discount $DICE_DISC \
--grad_update_type $GRAD_UPDATE_TYPE --learn_params $LEARN_PARAMS --chp_freq $CHP_FREQ --plot_freq $PLOT_FREQ \
--return_discount $RET_DISC --policy_type $POLICY_TYPE --load_chp $CHP --no_bias --visualize
