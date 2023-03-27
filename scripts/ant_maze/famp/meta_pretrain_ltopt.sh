#!/bin/bash

#SBATCH --job-name=ant642
#SBATCH --time=168:00:00
#SBATCH -N 3
#SBATCH --ntasks-per-node=16
#SBATCH --constraint=cpunode

. /etc/bashrc

EXPERIMENTS_DIR=...

# Algorithm settings
EPOCHS=10000 # Outer optimization EPOCHS
ENVS_PER_PROCESS=1 # Envs to be sampled per process (for maxspeed on cluster=1)
EPISODES=20 # Episodes per update step
LOOKAHEADS=2 # Inner updates to optimize for
GRAD_UPDATE_TYPE="meta"

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
LEARN_PARAMS="outer"

# Policy settings
POLICY_TYPE="ltopt"
OPTS=3
HID_OPT=(64 64)
HID_TERM=(64 64)
HID_SUBP=(64 64)


SEEDS=(642)
NCORES=48 # should be same as N * n-tasks-per-node

for SEED in "${SEEDS[@]}"; do
    RUN_NAME="seed${SEED}"
    RUN_DATA_DIR="${EXPERIMENTS_DIR}/ant_maze_famp/pretrain/"
    LOG_DIR="${RUN_DATA_DIR}/${RUN_NAME}/logs/"
    conda run -n famp mpiexec -np $NCORES -prepend-rank -outfile-pattern "${LOG_DIR}stdout.txt" -errfile-pattern "${LOG_DIR}stderr.txt" \
    python -u main.py --baseline $BASELINE --seed "$SEED" --env $ENV_NAME --run_name "$RUN_NAME" --run_data_dir ${RUN_DATA_DIR}  --epochs $EPOCHS --envs_per_process $ENVS_PER_PROCESS \
    --episodes $EPISODES --options $OPTS --gae_discount $GAE_DISC --lr_inner $LR_IN --lookaheads $LOOKAHEADS --dice_discount $DICE_DISC \
    --lr_outer $LR_OUT --grad_update_type $GRAD_UPDATE_TYPE --learn_params $LEARN_PARAMS\
    --return_discount $RET_DISC --policy_type $POLICY_TYPE --hidden_sizes_option ${HID_OPT[*]} \
    --hidden_sizes_termination ${HID_TERM[*]} --hidden_sizes_subpolicy ${HID_SUBP[*]} --learn_lr_inner
done
