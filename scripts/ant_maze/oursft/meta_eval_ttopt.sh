#!/bin/bash

#SBATCH --job-name=tte6_1
#SBATCH --time=168:00:00	
#SBATCH -N 1	
#SBATCH --ntasks-per-node=16	
#SBATCH --constraint=cpunode	
. /etc/bashrc	

EXPERIMENTS_DIR=...


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


SEEDS=(642)
#ENVS=(0 1 2 3 4)
ENVS=(5 6 7 8)
#ENVS=(9 10 11 12)
CHP_FREQ=10


for SEED in "${SEEDS[@]}"; do
    RUN_DATA_DIR="${EXPERIMENTS_DIR}/ant_maze_oursft/eval/seed${SEED}"
    CHP="${EXPERIMENTS_DIR}/ant_maze_oursft/pretrain/seed${SEED}/checkpoints/epoch2500.tar"
    for ENV in "${ENVS[@]}"; do
        RUN_NAME="env${ENV}"
        LOG_DIR="${RUN_DATA_DIR}/ant_maze_oursft/eval/seed${SEED}/logs/"
        conda run -n famp python -u main.py --baseline $BASELINE --seed "$SEED" --env $ENV_NAME --run_name "$RUN_NAME" \
        --run_data_dir $RUN_DATA_DIR   --epochs $EPOCHS --envs_per_process $ENVS_PER_PROCESS --episodes $EPISODES \
        --options $OPTS --gae_discount $GAE_DISC --lr_inner $LR_IN --lookaheads $LOOKAHEADS --dice_discount $DICE_DISC \
        --grad_update_type $GRAD_UPDATE_TYPE --learn_params $LEARN_PARAMS --return_discount $RET_DISC \
        --policy_type $POLICY_TYPE --hidden_sizes_option ${HID_OPT[*]} --term_time $TERM_TIME --load_chp $CHP\
        --hidden_sizes_subpolicy ${HID_SUBP[*]} --chp_freq $CHP_FREQ --fixed_env $ENV --learn_lr_inner --save_trajs
    done
done
