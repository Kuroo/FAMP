#!/bin/bash
#SBATCH --job-name=enh678
#SBATCH --time=168:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=16

. /etc/bashrc
EXPERIMENTS_DIR=...

# Algorithm settings
EPOCHS=50 # Outer optimization EPOCHS
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
LR_OUT=10.0
LR_IN=10.0
LEARN_PARAMS="inner"

# Policy settings
POLICY_TYPE="ltopt"
OPTS=1

#Others
CHP_FREQ=500
PLOT_FREQ=100

SEEDS=(642 742 842)
ENVS=(0 4 8 13 17 21 26 30 34 39 43 47)

for SEED in "${SEEDS[@]}"; do
    RUN_DATA_DIR="${EXPERIMENTS_DIR}/taxi_no_hierarchy/eval/seed${SEED}"
    CHP="${EXPERIMENTS_DIR}/taxi_no_hierarchy/pretrain/seed${SEED}/checkpoints/epoch2000.tar"
    for ENV in "${ENVS[@]}"; do
        RUN_NAME="env${ENV}"
        conda run -n famp python -u main.py --baseline $BASELINE --seed $SEED --env $ENV_NAME --run_name $RUN_NAME --epochs $EPOCHS --envs_per_process $ENVS_PER_PROCESS --episodes $EPISODES --options $OPTS\
        --fixed_env $ENV --gae_discount $GAE_DISC --lr_inner $LR_IN --lr_outer $LR_OUT --lookaheads $LOOKAHEADS --dice_discount $DICE_DISC --entropy_reg 0 --grad_update_type $GRAD_UPDATE_TYPE --return_discount $RET_DISC \
        --load_chp $CHP --learn_params $LEARN_PARAMS --plot_freq $PLOT_FREQ --chp_freq $CHP_FREQ --run_data_dir $RUN_DATA_DIR --no_bias --adapt_options
    done
done

