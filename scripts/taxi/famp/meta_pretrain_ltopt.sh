#!/bin/bash

#SBATCH --job-name=famp842
#SBATCH --time=168:00:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=16

. /etc/bashrc
EXPERIMENTS_DIR=...

# Algorithm settings
EPOCHS=2000 # Outer optimization EPOCHS
ENVS_PER_PROCESS=1 # Envs to be sampled per process (for maxspeed on cluster=1)
EPISODES=10 # Episodes per update step
LOOKAHEADS=3 # Inner updates to optimize for
GRAD_UPDATE_TYPE="meta"

# Env settings
ENV_NAME="taxi"
EXCLUDE_ENVS=(0 4 8 13 17 21 26 30 34 39 43 47)

# Discounts + Baseline settings
GAE_DISC=0.98
DICE_DISC=0
RET_DISC=0.95
BASELINE=linear

# Learning rate settings
LR_OUT=0.01
LR_IN=10.0
LEARN_PARAMS="outer"

# Policy settings
POLICY_TYPE="ltopt"
OPTS=4

#Others
CHP_FREQ=50
PLOT_FREQ=100

SEEDS=(842)
# SEEDS=(42 142 242 342 442)
NCORES=64 # should be same as N * n-tasks-per-node

for SEED in "${SEEDS[@]}"; do
    RUN_NAME="seed${SEED}"
    LOG_DIR="${EXPERIMENTS_DIR}/taxi_famp/pretrain/${RUN_NAME}/logs/"
    conda run -n famp mpiexec -np $NCORES -prepend-rank -outfile-pattern "${LOG_DIR}stdout.txt" -errfile-pattern "${LOG_DIR}stderr.txt" \
    python -u main.py --baseline $BASELINE --seed "$SEED" --env $ENV_NAME --run_name "$RUN_NAME" --run_data_dir "${EXPERIMENTS_DIR}/taxi_famp/pretrain" --epochs $EPOCHS --envs_per_process $ENVS_PER_PROCESS \
    --episodes $EPISODES --options $OPTS --gae_discount $GAE_DISC --lr_inner $LR_IN --lookaheads $LOOKAHEADS --dice_discount $DICE_DISC \
    --lr_outer $LR_OUT --grad_update_type $GRAD_UPDATE_TYPE --learn_params $LEARN_PARAMS --chp_freq $CHP_FREQ --plot_freq $PLOT_FREQ \
    --exclude_envs ${EXCLUDE_ENVS[*]} --return_discount $RET_DISC --policy_type $POLICY_TYPE --no_bias --termination_prior 0.5
done
