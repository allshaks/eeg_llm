#!/bin/bash -l
#
# log files (-o output, -e errors), not created automatically.
#SBATCH -o logs/%j_gpu1_undistr.out.undistr
#SBATCH -e logs/%j_gpu1_undistr.err.undistr
# -D is working directory.
#SBATCH -D ./
# -J is the name of the job. when running multiple jobs, helps check how long it takes, where it is in the queue, etc.
#SBATCH -J training_transformer_on_raw
#
# 
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --mem=30000

#
# only gpu nodes
# kind of gpu (a100 is the name, 1 is the number of gpus you request, in this case we don't use more)
#
# if I want to receive notification. "end" means sends notification _to_ the email. you can also set it to "none".
#SBATCH --mail-type=end
#SBATCH --mail-user=kammnoel1@gmail.com
#
# Wall clock limit (max. is 24 hours), if you need more time you would need to find a way to resume the job:
#SBATCH --time=02:00:00


source /etc/profile.d/modules.sh
# remove all modules that are already loaded
module purge


module load anaconda/3/2023.03 scikit-learn/1.2.2 pytorch/cpu/2.1.0 huggingface-transformers/4.31.0 
source ${HOME}/venvs/eeg_llm/bin/activate
pip install transformers 
pip install torch 
pip install scikit-learn
pip install numpy 
pip install pandas 
pip install matplotlib 

# pip install --user -e .
# pip install --user -r ./docs/requirements.txt
# pip install --user gdown

# --- TF related flags (provided by Tim) ---

# Avoid CUPTI warning message
# export LD_LIBRARY_PATH=${CUDA_HOME}/extras/CUPTI/lib64:${LD_LIBRARY_PATH}

# # Avoid OOM
# export TF_FORCE_GPU_ALLOW_GROWTH=true

# ## XLA
# # cuda aware
# export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_HOME}"
# # enable autoclustering for CPU and GPU
# export TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"

srun python transformer_on_toy_data.py