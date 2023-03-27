#!/bin/bash

#SBATCH --job-name=la8_6
#SBATCH --time=168:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH --constraint=cpunode

. /etc/bashrc

EXPERIMENTS_DIR=...

# Algorithm settings
EPOCHS=11 # Outer optimization EPOCHS
ENVS_PER_PROCESS=1 # Envs to be sampled per process (for maxspeed on cluster=1)
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
LR_INNER=0.0001
LR_OUT=0.0001
LEARN_PARAMS="inner"

# Policy settings
POLICY_TYPE="ltopt"
OPTS=3
HID_OPT=(64 64)
HID_TERM=(64 64)
HID_SUBP=(64 64)


SEEDS=(842)
ENVS=(6)
#ENVS=(0 1 2 3 4)
#ENVS=(5 6 7 8 9)
#ENVS=(10 11 12)

NCORES=16 # should be same as N * n-tasks-per-node

for SEED in "${SEEDS[@]}"; do
    RUN_DATA_DIR="${EXPERIMENTS_DIR}/ant_maze_learn_all/eval/seed${SEED}"
    CHP="${EXPERIMENTS_DIR}/ant_maze_learn_all/pretrain/seed${SEED}/checkpoints/epoch2500.tar"
    for ENV in "${ENVS[@]}"; do
        RUN_NAME="env${ENV}"
        LOG_DIR="${RUN_DATA_DIR}/ant_maze_learn_all/eval/seed${SEED}/logs/"
        conda run -n famp python -u main.py --baseline $BASELINE --seed $SEED --env $ENV_NAME --run_name "$RUN_NAME" --epochs $EPOCHS --envs_per_process $ENVS_PER_PROCESS --episodes $EPISODES --options $OPTS\
        --fixed_env $ENV --gae_discount $GAE_DISC --lr_inner $LR_INNER --lookaheads $LOOKAHEADS --dice_discount $DICE_DISC --entropy_reg 0 --grad_update_type $GRAD_UPDATE_TYPE --return_discount $RET_DISC \
        --load_chp $CHP --learn_params $LEARN_PARAMS --chp_freq 10 --run_data_dir $RUN_DATA_DIR --policy_type $POLICY_TYPE --hidden_sizes_option ${HID_OPT[*]} \
    --hidden_sizes_termination ${HID_TERM[*]} --hidden_sizes_subpolicy ${HID_SUBP[*]} --learn_lr_inner --adapt_options
    done
done
