#!/bin/bash
#SBATCH --job-name=single742
#SBATCH --time=168:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=16

. /etc/bashrc

EXPERIMENTS_DIR=...

seeds=(742)
#seeds=(42 142 242 342 442)
opts=4
lookaheads=0
grad_update_type="single"
env_name="taxi"
baseline="linear"
#envs=(0)
envs=(0 4 8 13 17 21 26 30 34 39 43 47)

for seed in "${seeds[@]}"; do
    for env in "${envs[@]}"; do
        run_name=${env}
        conda run -n famp python -u main.py --baseline $baseline --seed $seed --env $env_name --run_name ${run_name} --run_data_dir "${EXPERIMENTS_DIR}/taxi_single/eval/seed${seed}"  --epochs 500 --envs_per_process 1 --episodes 10 --options $opts --gae_discount 0.98 --lr_outer 0.3 --lookaheads $lookaheads --dice_discount 0 --grad_update_type $grad_update_type --termination_prior 0.5 --no_bias --learn_params all --chp_freq 500 --random_move_prob 0 --plot_freq 5000 --return_discount 0.95 --max_option_prob 1 --fixed_env $env
    done
done

