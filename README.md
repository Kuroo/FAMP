# Reusable Options through Gradient-based Meta Learning

This is the code that was used for the TMLR paper Reusable Options through Gradient-based Meta Learning (https://openreview.net/forum?id=qdDmxzGuzu).

## Visualization scripts
Scripts to run several trained models are included in the repo. In order to show policies during evaluation one should:
1. Install conda env. In this dir
       conda env -f environment.yml
2. (Optional) Configure mujoco
3. Activate conda env
       conda activate famp
4. Go to code dir
       cd code
5. Run visualization script (replace ant_maze with taxi for taxi env and meata_lt for meta_tt for fixed terminations) 
       ./scripts/ant_maze/visualize/meta_lt.sh .

## Training models
Scripts that were used to train and evaluate algorithms in the paper can be found in the scripts directory. They should contain all hyperparameters and flags. The exact conda environment that was used for these runs was exported using conda and is included as *environment_cluster.yml*. 

The experiments were run on a cluster using slurm but it should be possible to use them on a single machine as well. I recommend decreasing *NCORES* parameter in that case and increasing *ENVS_PER_CORE*. Example:
1. Go to code dir
       cd code
2. Run a training/test script:
       ../scripts/ant_maze/famp/meta_pretrain_ltopt.sh
